import argparse
import os
from tqdm import tqdm
import sys
import numpy as np
from collections import deque

sys.path.append('.')

from train.data import AgentGraphDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
import torch
import torch.nn as nn
from train.models.defender.model import MyGAT
from datetime import datetime
import torch.nn.functional as F
from train.trainer import BucketSampler, collate_with_time_padding


def shortest_path_distance_from_sources(adj_matrix, sources):
    """Shortest-path distance from a set of source nodes to all nodes."""
    num_nodes = adj_matrix.shape[0]
    if len(sources) == 0:
        return np.full(num_nodes, np.inf)
    dist = np.full(num_nodes, np.inf)
    q = deque()
    for s in sources:
        dist[s] = 0
        q.append(s)
    while q:
        u = q.popleft()
        for v in range(num_nodes):
            if adj_matrix[u, v] > 0 and dist[v] == np.inf:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def adjust_predictions_by_relationship(p_mal_np, p_inf_np, edge_index_np, num_nodes, thresh_mal=0.5, thresh_inf=0.5, sp_far_threshold=2):
    """
    Adjust predictions based on the relationship between malicious and infected agents.
    This mirrors main_defense_repair_test.py but omits EMA (no temporal state here).
    """
    # Build adjacency matrix.
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        adj_matrix[src, dst] = 1
    
    # Build neighbor sets.
    neighbors = {i: set(np.where(adj_matrix[i] > 0)[0].tolist()) for i in range(num_nodes)}
    
    # Initial sets.
    mal_set = set(np.where(p_mal_np >= thresh_mal)[0].tolist())
    inf_set = set(np.where(p_inf_np >= thresh_inf)[0].tolist())
    
    # Infection handling heuristic.
    if len(inf_set) > 0:
        sp_dist = shortest_path_distance_from_sources(adj_matrix, mal_set)
        adjusted_inf = set()
        for i in inf_set:
            adjacent_mal = any((nbr in mal_set) for nbr in neighbors[i])
            if adjacent_mal:
                adjusted_inf.add(i)
                continue
            # Without EMA: if too far, drop; otherwise pull in the strongest malicious neighbor.
            if sp_dist[i] > sp_far_threshold or np.isinf(sp_dist[i]):
                continue
            if len(neighbors[i]) > 0:
                nbr_list = list(neighbors[i])
                j = int(nbr_list[np.argmax(p_mal_np[nbr_list])])
                mal_set.add(j)
            adjusted_inf.add(i)
        inf_set = adjusted_inf
    
    # Ensure infected does not include malicious.
    inf_set = inf_set - mal_set
    
    # Build adjusted predictions.
    adjusted_pred_mal = np.zeros(num_nodes, dtype=int)
    adjusted_pred_inf = np.zeros(num_nodes, dtype=int)
    for i in mal_set:
        adjusted_pred_mal[i] = 1
    for i in inf_set:
        adjusted_pred_inf[i] = 1
    
    return adjusted_pred_mal, adjusted_pred_inf


def test(model, test_loader, criterion, device, alpha=1.0, beta=0.5, is_related_relocate=False):
    """Test loop (kept consistent with trainer.py:test)."""
    model.eval()
    running_loss = 0.0
    correct_both = 0
    total = 0
    correct_mal = 0
    correct_inf = 0

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            x, y, edge_index, edge_attr = data.x.to(device), data.y.to(device), data.edge_index.to(device), data.edge_attr.to(device)
            max_t = edge_attr.shape[1]
            x = edge_attr[:, 0, :]
            x = scatter_mean(x, edge_index[1], dim=0, dim_size=len(data.x))

            # Node self-reply features (optional).
            node_self_replies = None
            if hasattr(data, 'node_self_replies') and data.node_self_replies is not None:
                node_self_replies = data.node_self_replies.to(device)

            outputs = model(x, edge_index, edge_attr, num_turns=max_t, node_self_replies=node_self_replies)
            bce_loss = criterion(outputs, y.float())
            probs = torch.sigmoid(outputs).to(device)
            p_mal = probs[:, 0]
            p_inf = probs[:, 1]
            
            # Optionally adjust predictions based on mal/inf relationships.
            if is_related_relocate:
                # Handle batched graphs by splitting on data.batch.
                batch = data.batch if hasattr(data, 'batch') else None
                if batch is not None:
                    # Process each sample in the batch independently.
                    adjusted_pred_mal_list = []
                    adjusted_pred_inf_list = []
                    unique_batches = torch.unique(batch)
                    
                    for batch_idx in unique_batches:
                        # Node indices for this sample (global indices).
                        sample_mask = (batch == batch_idx)
                        sample_nodes = torch.where(sample_mask)[0].cpu()
                        num_nodes_sample = len(sample_nodes)
                        
                        # Edge_index for this sample (global indices).
                        sample_edge_mask = torch.isin(edge_index[0].cpu(), sample_nodes) & torch.isin(edge_index[1].cpu(), sample_nodes)
                        sample_edge_index = edge_index[:, sample_edge_mask]
                        
                        # Map global node indices to local [0, num_nodes_sample).
                        node_map = {int(global_idx): local_idx for local_idx, global_idx in enumerate(sample_nodes)}
                        sample_edge_index_local = torch.zeros_like(sample_edge_index)
                        for i in range(sample_edge_index.shape[1]):
                            sample_edge_index_local[0, i] = node_map[int(sample_edge_index[0, i])]
                            sample_edge_index_local[1, i] = node_map[int(sample_edge_index[1, i])]
                        
                        # Per-sample probabilities.
                        p_mal_sample = p_mal[sample_mask].cpu().numpy()
                        p_inf_sample = p_inf[sample_mask].cpu().numpy()
                        edge_index_np = sample_edge_index_local.cpu().numpy()
                        
                        # Adjust predictions (local indices).
                        adj_pred_mal, adj_pred_inf = adjust_predictions_by_relationship(
                            p_mal_sample, p_inf_sample, edge_index_np, num_nodes_sample
                        )
                        
                        # Map back to global indices.
                        global_adj_pred_mal = np.zeros(len(data.x), dtype=int)
                        global_adj_pred_inf = np.zeros(len(data.x), dtype=int)
                        for local_idx, global_idx in enumerate(sample_nodes):
                            global_adj_pred_mal[global_idx] = adj_pred_mal[local_idx]
                            global_adj_pred_inf[global_idx] = adj_pred_inf[local_idx]
                        
                        adjusted_pred_mal_list.append(global_adj_pred_mal)
                        adjusted_pred_inf_list.append(global_adj_pred_inf)
                    
                    # Merge adjusted predictions across samples.
                    adjusted_pred_mal_all = np.zeros(len(data.x), dtype=int)
                    adjusted_pred_inf_all = np.zeros(len(data.x), dtype=int)
                    for pred_mal_arr, pred_inf_arr in zip(adjusted_pred_mal_list, adjusted_pred_inf_list):
                        adjusted_pred_mal_all = np.maximum(adjusted_pred_mal_all, pred_mal_arr.astype(int))
                        adjusted_pred_inf_all = np.maximum(adjusted_pred_inf_all, pred_inf_arr.astype(int))
                    pred_mal = torch.tensor(adjusted_pred_mal_all, device=device, dtype=torch.long)
                    pred_inf = torch.tensor(adjusted_pred_inf_all, device=device, dtype=torch.long)
                else:
                    # No batch info; assume a single graph.
                    edge_index_np = edge_index.cpu().numpy()
                    p_mal_np = p_mal.cpu().numpy()
                    p_inf_np = p_inf.cpu().numpy()
                    num_nodes = len(data.x)
                    
                    adj_pred_mal, adj_pred_inf = adjust_predictions_by_relationship(
                        p_mal_np, p_inf_np, edge_index_np, num_nodes
                    )
                    pred_mal = torch.tensor(adj_pred_mal, device=device, dtype=torch.long)
                    pred_inf = torch.tensor(adj_pred_inf, device=device, dtype=torch.long)
                
                # Use adjusted predictions.
                pred = torch.stack([pred_mal, pred_inf], dim=1)
            else:
                # Use raw predictions.
                pred = (probs >= 0.5).long()
            
            from torch_scatter import scatter_max
            src, dst = data.edge_index[0].to(device), data.edge_index[1].to(device)
            max_vals, _ = scatter_max(p_mal[src], dst, dim=0, dim_size=p_mal.size(0))
            max_vals[max_vals == float('-inf')] = 0.0
            consistency_loss = (p_inf * (1 - max_vals) ** 2).mean()
            loss = bce_loss[:, 0].mean() + alpha * bce_loss[:, 1].mean() + beta * consistency_loss
            running_loss += loss.item()

            total += y.size(0)
            correct_mal += (pred[:, 0] == y[:, 0].long()).sum().item()
            correct_inf += (pred[:, 1] == y[:, 1].long()).sum().item()
            correct_both += ((pred[:, 0] == y[:, 0].long()) & (pred[:, 1] == y[:, 1].long())).sum().item()

    avg_loss = running_loss / len(test_loader)
    acc_both = 100 * correct_both / total
    acc_mal = 100 * correct_mal / total
    acc_inf = 100 * correct_inf / total

    return avg_loss, acc_both, acc_mal, acc_inf


def find_latest_model(checkpoint_dir, dataset):
    """Auto-find the latest model checkpoint."""
    # First, check latest_model_path.txt.
    latest_model_path_file = os.path.join(checkpoint_dir, dataset, "latest_model_path.txt")
    if os.path.exists(latest_model_path_file):
        with open(latest_model_path_file, 'r') as f:
            model_path = f.read().strip()
            if os.path.exists(model_path):
                print(f"Using model specified in latest_model_path.txt: {model_path}")
                return model_path
    
    # Otherwise, pick the newest .pth in the directory.
    dataset_dir = os.path.join(checkpoint_dir, dataset)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Directory does not exist: {dataset_dir}")
    
    pth_files = [f for f in os.listdir(dataset_dir) if f.endswith('.pth')]
    if not pth_files:
        raise FileNotFoundError(f"No .pth model files found under: {dataset_dir}")
    
    # Sort by mtime and pick the newest.
    pth_files.sort(key=lambda x: os.path.getmtime(os.path.join(dataset_dir, x)), reverse=True)
    latest_model = os.path.join(dataset_dir, pth_files[0])
    print(f"Using the newest model in the directory: {latest_model}")
    return latest_model


def parse_arguments():
    parser = argparse.ArgumentParser(description="Test trained GAT model")

    parser.add_argument("--attack_mode", type=str, required=True, choices=["PI", "MA", "TA"], 
                       help="Attack mode: PI (Prompt Injection), MA (Memory Attack), TA (Tool Attack)")
    parser.add_argument("--dataset", type=str, default="mmlu", choices=["mmlu", "csqa", "gsm8k", "memory_attack", "tool_attack"],
                       help="Dataset type for PI attack mode (required for PI, ignored for MA/TA)")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Custom dataset path (optional, will use default if not provided)")
    
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint",
                       help="Directory containing model checkpoints")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to specific model checkpoint (if not provided, will auto-find latest)")
    parser.add_argument("--test_phase", type=str, default="val", choices=["val", "test"],
                       help="Test phase: val (validation set) or test (test set)")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--is_related_relocate", action="store_true", default=False,
                       help="If True, adjust predictions based on relationship between malicious and infected agents")
    args = parser.parse_args()

    # Set default dataset paths based on attack_mode
    if args.dataset_path is None:
        if args.attack_mode == "PI":
            if args.dataset == "mmlu":
                args.dataset_path = f"./output/output_{args.name}/ModelTrainingSet_{args.name}/mmlu/dataset.pkl"
            elif args.dataset == "csqa":
                args.dataset_path = f"./output/output_{args.name}/ModelTrainingSet_{args.name}/csqa/dataset.pkl"
            elif args.dataset == "gsm8k":
                args.dataset_path = f"./output/output_{args.name}/ModelTrainingSet_{args.name}/gsm8k/dataset.pkl"
            else:
                raise Exception(f"Unknown dataset {args.dataset} for PI attack mode")
        elif args.attack_mode == "MA":
            args.dataset_path = f"./output/output_{args.name}/ModelTrainingSet_{args.name}/memory_attack/dataset.pkl"
        elif args.attack_mode == "TA":
            args.dataset_path = f"./output/output_{args.name}/ModelTrainingSet_{args.name}/tool_attack/dataset.pkl"
        else:
            raise Exception(f"Unknown attack mode {args.attack_mode}")

    # Determine dataset name for model loading
    if args.attack_mode == "PI":
        dataset_name = args.dataset
    else:
        normalized_path = os.path.normpath(args.dataset_path)
        parts = normalized_path.split(os.sep)
        dataset_name = parts[-2]

    # Auto-find model if not specified
    if args.model_path is None:
        args.model_path = find_latest_model(f'./output/output_{args.name}/{args.checkpoint_dir}_{args.name}', f'{dataset_name}')
    else:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Specified model file does not exist: {args.model_path}")
        print(f"Using specified model path: {args.model_path}")

    return args


def main():
    args = parse_arguments()
    
    # Load test dataset.
    print(f"Loading test dataset: {args.dataset_path}")
    test_dataset = AgentGraphDataset(args.dataset_path, phase=args.test_phase)
    
    # Use a global bucket sampler.
    test_sampler = BucketSampler(test_dataset.turns, batch_size=args.batch_size, shuffle=False, bucket_multiplier=8)
    testloader = TorchDataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, shuffle=False, collate_fn=collate_with_time_padding)
    
    # Model input dims.
    example = test_dataset[0]
    in_channels = example.x.size(1)
    edge_dim = example.edge_attr.size()[1:]
    
    # Build model.
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    gnn = MyGAT(
        in_channels=in_channels, 
        hidden_channels=args.hidden_dim, 
        out_channels=2, 
        heads=args.num_heads, 
        num_layers=args.num_layers, 
        edge_dim=edge_dim,
        dropout=args.dropout
    )
    gnn.to(device)
    
    # Load weights.
    print(f"Loading weights: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    gnn.load_state_dict(state_dict)
    print("Weights loaded.")
    
    # Eval mode.
    gnn.eval()
    
    # Loss.
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    # Run test.
    print(f"\nTesting attack_mode={args.attack_mode}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Dataset size: {len(test_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model: hidden_dim={args.hidden_dim}, num_heads={args.num_heads}, num_layers={args.num_layers}")
    print(f"Loss weights: alpha={args.alpha}, beta={args.beta}")
    print(f"Relationship adjustment: is_related_relocate={args.is_related_relocate}")
    print("-" * 80)
    
    test_loss, test_acc_both, test_acc_mal, test_acc_inf = test(
        gnn, testloader, criterion, device=device, alpha=args.alpha, beta=args.beta, is_related_relocate=args.is_related_relocate
    )
    
    # Results.
    print("-" * 80)
    print("Test results:")
    print(f"  Total Loss: {test_loss:.4f}")
    print(f"  Accuracy (both): {test_acc_both:.2f}%")
    print(f"  Accuracy (malicious): {test_acc_mal:.2f}%")
    print(f"  Accuracy (infected): {test_acc_inf:.2f}%")
    print("-" * 80)
    
    # Save results to file.
    result_dir = os.path.dirname(args.model_path)
    result_file = os.path.join(result_dir, f"test_results_{args.test_phase}.txt")
    with open(result_file, 'w') as f:
        f.write(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model path: {args.model_path}\n")
        f.write(f"Dataset path: {args.dataset_path}\n")
        f.write(f"Test phase: {args.test_phase}\n")
        f.write(f"Attack mode: {args.attack_mode}\n")
        f.write(f"Dataset name: {args.dataset}\n")
        f.write(f"Model params: hidden_dim={args.hidden_dim}, num_heads={args.num_heads}, num_layers={args.num_layers}\n")
        f.write(f"Loss weights: alpha={args.alpha}, beta={args.beta}\n")
        f.write(f"Relationship adjustment: is_related_relocate={args.is_related_relocate}\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total loss: {test_loss:.4f}\n")
        f.write(f"Accuracy (both): {test_acc_both:.2f}%\n")
        f.write(f"Accuracy (malicious): {test_acc_mal:.2f}%\n")
        f.write(f"Accuracy (infected): {test_acc_inf:.2f}%\n")
    
    print(f"Saved test results to: {result_file}")


if __name__ == "__main__":
    main()

