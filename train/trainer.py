import argparse
import os
from tqdm import tqdm
import sys

sys.path.append('.')

from train.data import AgentGraphDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW 
from torch.optim.lr_scheduler import CosineAnnealingLR
from train.models.defender.model import MyGAT
from einops import rearrange
from datetime import datetime
import random 
import torch.nn.functional as F
from torch.utils.data import Sampler

class BucketSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True, bucket_multiplier=2):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = max(batch_size * bucket_multiplier, batch_size)
        self.indices = list(range(len(self.lengths)))
    def __iter__(self):
        idxs = self.indices
        idxs.sort(key=lambda i: self.lengths[i])
        # Split into buckets; optionally shuffle bucket order and in-bucket order.
        buckets = [idxs[i:i+self.bucket_size] for i in range(0, len(idxs), self.bucket_size)]
        if self.shuffle:
            random.shuffle(buckets)
        ordered = []
        for b in buckets:
            if self.shuffle:
                random.shuffle(b)
            ordered.extend(b)
        return iter(ordered)
    def __len__(self):
        return len(self.lengths)

def pad_edge_attr(edge_attr, target_t):
    # edge_attr shape: (num_edges, T, D)
    T = edge_attr.shape[1]
    pad_len = target_t - T
    if pad_len > 0:
        # Pad on the right along the T axis.
        return F.pad(edge_attr, (0, 0, 0, pad_len, 0, 0))
    else:
        return edge_attr

def collate_with_time_padding(batch_list):
    # Compute max T in this batch.
    max_t = 0
    for d in batch_list:
        t = d.edge_attr.shape[1]
        if t > max_t:
            max_t = t
    # Right-pad edge_attr/node_self_replies/infection labels along T to max_t.
    padded = []
    for d in batch_list:
        ea = d.edge_attr
        T = ea.shape[1]
        if T < max_t:
            ea = F.pad(ea, (0, 0, 0, max_t - T, 0, 0))
            d.edge_attr = ea
        # Align node_self_replies along the time dimension if present.
        if hasattr(d, 'node_self_replies') and d.node_self_replies is not None:
            nsr = d.node_self_replies
            if nsr.shape[1] < max_t:
                nsr = F.pad(nsr, (0, 0, 0, max_t - nsr.shape[1], 0, 0))
                d.node_self_replies = nsr
        if hasattr(d, 'inf_labels_per_turn') and d.inf_labels_per_turn is not None:
            inf = d.inf_labels_per_turn
            if inf.shape[1] < max_t:
                inf = F.pad(inf, (0, max_t - inf.shape[1], 0, 0))
            elif inf.shape[1] > max_t:
                inf = inf[:, :max_t]
            d.inf_labels_per_turn = inf
        padded.append(d)
    # Merge via PyG Batch.
    return Batch.from_data_list(padded)

def train_gsafeguard(model: MyGAT, train_loader, criterion, optimizer, device, attack_mode):
    """Training loop for the GSAFeguard variant (single-label output)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data in train_loader:
        x, y, edge_index, edge_attr = data.x.to(device), data.y.to(device), data.edge_index.to(device), data.edge_attr.to(device)
        x = edge_attr[:, 0, :]
        x = scatter_mean(x, edge_index[0], dim=0, dim_size=len(data.x))
        
        # Apply random turns for MA and TA attack modes
        if attack_mode in ["MA", "TA"]: # [TODO] gsafeguard only MA TA
            random_turns = random.choice(list(range(1, 5)))
            edge_attr = edge_attr[:, :random_turns, :]
            
        optimizer.zero_grad()
        outputs = model(x, edge_index=edge_index, edge_attr=edge_attr)
        
        loss = criterion(outputs, y[:, 0].float().unsqueeze(-1))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predicted = (torch.sigmoid(outputs) >= 0.5).squeeze()
        total += y[:, 0].size(0)
        correct += (predicted == y[:, 0]).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def test_gsafeguard(model, test_loader, criterion, device):
    """Evaluation loop for the GSAFeguard variant (single-label output)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            x, y, edge_index, edge_attr = data.x.to(device), data.y.to(device), data.edge_index.to(device), data.edge_attr.to(device)
            x = edge_attr[:, 0, :]
            x = scatter_mean(x, edge_index[1], dim=0, dim_size=len(data.x))

            outputs = model(x, edge_index, edge_attr)
            loss = criterion(outputs, y[:, 0].float().unsqueeze(-1))
            running_loss += loss.item()

            predicted = (torch.sigmoid(outputs) >= 0.5).squeeze()
            total += y[:, 0].size(0)
            correct += (predicted == y[:, 0]).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def train(model: MyGAT, train_loader, criterion, optimizer, device, attack_mode, alpha=0.5, beta=0.5, selective_training=True):
    """
    selective_training: If True, only train the branch for the sampled turn-range; otherwise train all parameters.
    """
    model.train()
    running_loss = 0.0
    correct_both = 0
    total = 0
    correct_mal = 0
    correct_inf = 0

    for data in train_loader:
        x, y, edge_index, edge_attr = data.x.to(device), data.y.to(device), data.edge_index.to(device), data.edge_attr.to(device)
        x = edge_attr[:, 0, :]
        x = scatter_mean(x, edge_index[0], dim=0, dim_size=len(data.x))
        
        # edge_attr is already padded in collate: [num_edges_total, T, D]
        max_t = edge_attr.shape[1]
        
        # Node self-reply features (optional).
        node_self_replies = None
        if hasattr(data, 'node_self_replies') and data.node_self_replies is not None:
            node_self_replies = data.node_self_replies.to(device)
        
        # Apply random turns for MA and TA attack modes
        actual_turns = max_t
        if attack_mode in ["MA", "TA", "PI"]:
            random_turns = random.choice(list(range(1, max_t + 1)))
            edge_attr = edge_attr[:, :random_turns, :]
            if node_self_replies is not None:
                node_self_replies = node_self_replies[:, :random_turns, :]
            actual_turns = random_turns
        
        # Select infection labels for the sampled number of turns (if available).
        if hasattr(data, 'inf_labels_per_turn') and data.inf_labels_per_turn is not None:
            inf_labels_per_turn = data.inf_labels_per_turn.to(device)
            total_turns_available = inf_labels_per_turn.shape[1]
            if total_turns_available > 0:
                turn_idx = min(actual_turns, total_turns_available) - 1
                if turn_idx >= 0:
                    y = y.clone()
                    y[:, 1] = inf_labels_per_turn[:, turn_idx]
        
        # Selective training: freeze other branches.
        if selective_training:
            # Select the branch for the sampled turns.
            branch_idx = model._select_branch(actual_turns)
            # Freeze all parameters.
            for param in model.parameters():
                param.requires_grad = False
            # Unfreeze shared layers and the selected branch.
            for param in model.shared_convs.parameters():
                param.requires_grad = True
            for param in model.diag_emb_proc.parameters():
                param.requires_grad = True
            if hasattr(model, 'node_self_reply_proc'):
                for param in model.node_self_reply_proc.parameters():
                    param.requires_grad = True
            if branch_idx < len(model.branch_convs):
                for param in model.branch_convs[branch_idx].parameters():
                    param.requires_grad = True
                for param in model.branch_mlps[branch_idx].parameters():
                    param.requires_grad = True
                for param in model.branch_inf_mlps[branch_idx].parameters():
                    param.requires_grad = True
                for param in model.branch_heads_mal[branch_idx].parameters():
                    param.requires_grad = True
                for param in model.branch_heads_inf[branch_idx].parameters():
                    param.requires_grad = True
        
        optimizer.zero_grad()
        outputs = model(x, edge_index=edge_index, edge_attr=edge_attr, num_turns=actual_turns, node_self_replies=node_self_replies)
        # Multi-label BCE.
        bce_loss = criterion(outputs, y.float())

        # Probabilities.
        probs = torch.sigmoid(outputs)
        p_mal = probs[:, 0]
        p_inf = probs[:, 1]

        # Consistency loss: 1-hop neighbor approximation.
        # For each edge u->v, propagate p_mal[u] to v and take max over incoming neighbors.
        from torch_scatter import scatter_max
        src, dst = edge_index[0], edge_index[1]

        neigh_max_mal = torch.zeros_like(p_mal)
        max_vals, _ = scatter_max(p_mal[src], dst, dim=0, dim_size=p_mal.size(0))
        # scatter_max uses -inf for nodes with no incoming edges; replace with 0.
        max_vals[max_vals == float('-inf')] = 0.0
        neigh_max_mal = max_vals

        neigh_max_inf = torch.zeros_like(p_inf)
        max_vals_inf, _ = scatter_max(p_inf[src], dst, dim=0, dim_size=p_inf.size(0))
        # scatter_max uses -inf for nodes with no incoming edges; replace with 0.
        max_vals_inf[max_vals_inf == float('-inf')] = 0.0
        neigh_max_inf = max_vals_inf


        consistency_loss = (p_inf * (1 - neigh_max_mal) ** 2 * (1 - neigh_max_inf) ** 2).mean()

        loss = bce_loss[:, 0].mean() + alpha * bce_loss[:, 1].mean() + beta * consistency_loss

        loss.backward()
        optimizer.step()
        
        # Restore requires_grad for all params (for the next batch).
        if selective_training:
            for param in model.parameters():
                param.requires_grad = True

        running_loss += loss.item()

        pred = (probs >= 0.5).long()
        total += y.size(0)
        correct_mal += (pred[:, 0] == y[:, 0].long()).sum().item()
        correct_inf += (pred[:, 1] == y[:, 1].long()).sum().item()
        correct_both += ((pred[:, 0] == y[:, 0].long()) & (pred[:, 1] == y[:, 1].long())).sum().item()

    avg_loss = running_loss / len(train_loader)
    acc_both = 100 * correct_both / total
    acc_mal = 100 * correct_mal / total
    acc_inf = 100 * correct_inf / total

    return avg_loss, acc_both, acc_mal, acc_inf


def test(model, test_loader, criterion, device, alpha=1.0, beta=0.5, use_branch_aggregation=False):
    """
    use_branch_aggregation: If True, average outputs across all branches; otherwise use the branch for the current number of turns.
    """
    model.eval()
    running_loss = 0.0
    correct_both = 0
    total = 0
    correct_mal = 0
    correct_inf = 0

    with torch.no_grad():
        for data in test_loader:
            x, y, edge_index, edge_attr = data.x.to(device), data.y.to(device), data.edge_index.to(device), data.edge_attr.to(device)
            max_t = edge_attr.shape[1]
            x = edge_attr[:, 0, :]
            x = scatter_mean(x, edge_index[1], dim=0, dim_size=len(data.x))

            # Node self-reply features (optional).
            node_self_replies = None
            if hasattr(data, 'node_self_replies') and data.node_self_replies is not None:
                node_self_replies = data.node_self_replies.to(device)

            if hasattr(data, 'inf_labels_per_turn') and data.inf_labels_per_turn is not None:
                inf_labels_per_turn = data.inf_labels_per_turn.to(device)
                total_turns_available = inf_labels_per_turn.shape[1]
                if total_turns_available > 0:
                    if use_branch_aggregation:
                        turn_idx = total_turns_available - 1
                    else:
                        turn_idx = min(max_t, total_turns_available) - 1
                    if turn_idx >= 0:
                        y = y.clone()
                        y[:, 1] = inf_labels_per_turn[:, turn_idx]

            # Inference: aggregate all branches if use_branch_aggregation=True; otherwise use the branch for max_t.
            if use_branch_aggregation:
                outputs = model(x, edge_index, edge_attr, num_turns=None, node_self_replies=node_self_replies)
            else:
                outputs = model(x, edge_index, edge_attr, num_turns=max_t, node_self_replies=node_self_replies)
            
            bce_loss = criterion(outputs, y.float())
            probs = torch.sigmoid(outputs).to(device)
            p_mal = probs[:, 0]
            p_inf = probs[:, 1]
            from torch_scatter import scatter_max
            src, dst = data.edge_index[0].to(device), data.edge_index[1].to(device)
            max_vals, _ = scatter_max(p_mal[src], dst, dim=0, dim_size=p_mal.size(0))
            max_vals[max_vals == float('-inf')] = 0.0
            consistency_loss = (p_inf * (1 - max_vals) ** 2).mean()
            loss = bce_loss[:, 0].mean() + alpha * bce_loss[:, 1].mean() + beta * consistency_loss
            # print(bce_loss[:, 0].mean(), bce_loss[:, 1].mean(), consistency_loss)
            running_loss += loss.item()

            pred = (probs >= 0.5).long()
            total += y.size(0)
            correct_mal += (pred[:, 0] == y[:, 0].long()).sum().item()
            correct_inf += (pred[:, 1] == y[:, 1].long()).sum().item()
            correct_both += ((pred[:, 0] == y[:, 0].long()) & (pred[:, 1] == y[:, 1].long())).sum().item()

    avg_loss = running_loss / len(test_loader)
    acc_both = 100 * correct_both / total
    acc_mal = 100 * correct_mal / total
    acc_inf = 100 * correct_inf / total

    return avg_loss, acc_both, acc_mal, acc_inf


def parse_arguments():
    parser = argparse.ArgumentParser(description="Experiments to train GAT")

    parser.add_argument("--attack_mode", type=str, required=True, choices=["PI", "MA", "TA"], 
                       help="Attack mode: PI (Prompt Injection), MA (Memory Attack), TA (Tool Attack)")
    parser.add_argument("--dataset", type=str, default="mmlu", choices=["mmlu", "csqa", "gsm8k", 'memory_attack', 'tool_attack'],
                       help="Dataset type for PI attack mode (required for PI, ignored for MA/TA)")
    parser.add_argument("--dataset_path", type=str, default=None,
                       help="Custom dataset path (optional, will use default if not provided)")
    
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)

    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs (default: 20 for PI/MA, 50 for TA)")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0002)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--save_dir", type=str, default="./checkpoint")
    parser.add_argument("--name", type=str, default="", help="Custom name suffix for output directories (prevents overwriting)")
    parser.add_argument("--selective_training", action="store_true", help="Enable selective training (only train corresponding branch)")
    parser.add_argument("--use_branch_aggregation", action="store_true", default=True, help="Use branch aggregation in test (default: True)")
    parser.add_argument("--guard", type=str, default="ours", choices=["gsafeguard", "ours"],
                       help="Guard version: gsafeguard (single-label) or ours (dual-label, default)")

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

    # Set default epochs based on attack_mode
    if args.epochs is None:
        if args.attack_mode == "TA":
            args.epochs = 50
        else:
            args.epochs = 20

    # Set save directory based on attack_mode and dataset
    if args.attack_mode == "PI":
        dataset_name = args.dataset
    else:
        normalized_path = os.path.normpath(args.dataset_path)
        parts = normalized_path.split(os.sep)
        dataset_name = parts[-2]
    
    # Add name suffix if provided
    if args.name:
        dataset_name = f"{dataset_name}"
    
    args.save_dir = os.path.join(args.save_dir, dataset_name, args.guard)
    if not os.path.exists(args.save_dir): 
        os.makedirs(args.save_dir)

    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{current_time_str}-hiddim_{args.hidden_dim}-heads_{args.num_heads}-layers_{args.num_layers}-epochs_{args.epochs}-lr_{args.lr}-dropout_{args.dropout}-wd_{args.weight_decay}.pth"
    args.save_path = os.path.join(args.save_dir, filename)

    return args


def main():
    args = parse_arguments()
    
    train_dataset = AgentGraphDataset(args.dataset_path, phase="train")
    val_dataset = AgentGraphDataset(args.dataset_path, phase="val")
    example = train_dataset[0]
    in_channels = example.x.size(1)
    edge_dim = example.edge_attr.size()[1:]
    
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    
    # Choose output channels based on the guard variant.
    if args.guard == "gsafeguard":
        out_channels = 1
        # GSAFeguard uses a simple DataLoader.
        from torch_geometric.loader import DataLoader as PyGDataLoader
        trainloader = PyGDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        testloader = PyGDataLoader(val_dataset)
    else:
        out_channels = 2
        # Our variant uses a bucket sampler and a custom collate_fn.
        train_sampler = BucketSampler(train_dataset.turns, batch_size=args.batch_size, shuffle=True, bucket_multiplier=8)
        val_sampler = BucketSampler(val_dataset.turns, batch_size=args.batch_size, shuffle=False, bucket_multiplier=8)
        trainloader = TorchDataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=False, collate_fn=collate_with_time_padding)
        testloader = TorchDataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, shuffle=False, collate_fn=collate_with_time_padding)
    
    gnn = MyGAT(in_channels, args.hidden_dim, out_channels=out_channels, heads=args.num_heads, num_layers=args.num_layers, edge_dim=edge_dim, guard=args.guard)
    gnn.to(device)
    if args.guard == "ours":
        criterion = nn.BCEWithLogitsLoss(reduction='none')  
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
    best_acc = 0.0
    best_inf_acc = 0.0
    print(f"Starting training for {args.attack_mode} attack mode (guard={args.guard})")
    print(f"Dataset: {args.dataset_path}")
    print(f"Training parameters: epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}")
    
    for i in range(args.epochs): 
        if args.guard == "gsafeguard":
            train_loss, train_acc = train_gsafeguard(gnn, trainloader, criterion, optimizer, device=device, attack_mode=args.attack_mode)
            test_loss, test_acc = test_gsafeguard(gnn, testloader, criterion, device=device)
            scheduler.step()
            if test_acc > best_acc: 
                best_acc = test_acc
                torch.save(gnn.state_dict(), args.save_path)
                print(f"Epoch {i}/{args.epochs} || Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}% || Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}% || Save!")
            else:
                print(f"Epoch {i}/{args.epochs} || Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}% || Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
        else:
            train_loss, train_acc_both, train_acc_mal, train_acc_inf = train(gnn, trainloader, criterion, optimizer, device=device, attack_mode=args.attack_mode, alpha=args.alpha, beta=args.beta, selective_training=args.selective_training)
            test_loss, test_acc_both, test_acc_mal, test_acc_inf = test(gnn, testloader, criterion, device=device, alpha=args.alpha, beta=args.beta)
            scheduler.step()
            if (test_acc_mal + test_acc_inf) > (best_acc + best_inf_acc): 
                best_acc = test_acc_mal
                best_inf_acc = test_acc_inf
                torch.save(gnn.state_dict(), args.save_path)
                print(f"Epoch {i}/{args.epochs} || Train Loss: {train_loss:.4f}, Acc(both/mal/inf): {train_acc_both:.2f}/{train_acc_mal:.2f}/{train_acc_inf:.2f}% || Test Loss: {test_loss:.4f}, Acc(both/mal/inf): {test_acc_both:.2f}/{test_acc_mal:.2f}/{test_acc_inf:.2f}% || Save!")
            
            else:
                print(f"Epoch {i}/{args.epochs} || Train Loss: {train_loss:.4f}, Acc(both/mal/inf): {train_acc_both:.2f}/{train_acc_mal:.2f}/{train_acc_inf:.2f}% || Test Loss: {test_loss:.4f}, Acc(both/mal/inf): {test_acc_both:.2f}/{test_acc_mal:.2f}/{test_acc_inf:.2f}%")
    
    print(f"Training completed. Best test accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {args.save_path}")
    
    # Save model path to a file for automated pipeline
    model_path_file = os.path.join(args.save_dir, "latest_model_path.txt")
    with open(model_path_file, 'w') as f:
        f.write(args.save_path)
    print(f"Model path saved to: {model_path_file}")


if __name__ == "__main__":
    main()
