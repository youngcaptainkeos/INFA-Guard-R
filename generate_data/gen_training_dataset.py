import os 
import json
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse


def gen_model_training_set(language_dataset, embedding_model, save_path, attack_mode="PI"): 
    """
    Generate the model training dataset.
    
    Args:
        language_dataset: Language dataset
        embedding_model: Embedding model
        save_path: Output path
        attack_mode: Attack mode ("PI", "MA", "TA")
    """
    dataset = []
    for meta_data in tqdm(language_dataset, desc=f"Generate training data for {attack_mode}"): 
        adj_matrix = meta_data["adj_matrix"]
        attacker_idxes = meta_data["attacker_idxes"]
        system_prompts = meta_data["system_prompts"]
        communication_data = meta_data["communication_data"]

        # PI has extra fields.
        if attack_mode == "PI":
            question = meta_data["question"]
            correct_answer = meta_data["correct_answer"]
            wrong_answer = meta_data["wrong_answer"]
        
        adj_matrix_np = np.array(adj_matrix)
        mal_labels = np.array([1 if i in attacker_idxes else 0 for i in range(len(adj_matrix)) ])
        system_prompts_embedding = []
        for i in range(len(system_prompts)): 
            system_prompts_embedding.append(embedding_model.encode(system_prompts[i]))
        system_prompts_embedding = np.array(system_prompts_embedding)

        # edge_embedding
        edge_index = adj_matrix_np.nonzero()
        edge_index = np.array(edge_index)
        communication_embeddings = [[] for _ in range(len(adj_matrix))]
        for i in range(len(communication_data)):
            turn_i_data = communication_data[i]
            turn_i_embeddings = [None] * len(turn_i_data)
            for agent_idx, c_data in turn_i_data: 
                i_turns_agent_idx_embedding = embedding_model.encode(c_data)
                turn_i_embeddings[agent_idx] = i_turns_agent_idx_embedding
            for agent_idx in range(len(turn_i_embeddings)): 
                communication_embeddings[agent_idx].append(turn_i_embeddings[agent_idx])
        
        communication_embeddings = np.array(communication_embeddings)

        # Build infection labels from per-turn infection indices produced by gen_graph.py.
        infected_idxes_per_turn = meta_data.get("infected_idxes_per_turn")
        infected_idxes = meta_data.get("infected_idxes", [])

        num_nodes = len(adj_matrix)
        num_turns = 0
        if communication_embeddings.ndim >= 2:
            num_turns = communication_embeddings.shape[1]

        if infected_idxes_per_turn is None:
            infected_idxes_per_turn = []
        else:
            infected_idxes_per_turn = [list(turn_idxes) for turn_idxes in infected_idxes_per_turn]

        if num_turns > 0:
            if len(infected_idxes_per_turn) == 0:
                infected_idxes_per_turn = [list(infected_idxes) for _ in range(num_turns)]
            elif len(infected_idxes_per_turn) < num_turns:
                last_turn = infected_idxes_per_turn[-1] if infected_idxes_per_turn else list(infected_idxes)
                infected_idxes_per_turn = infected_idxes_per_turn + [list(last_turn) for _ in range(num_turns - len(infected_idxes_per_turn))]
            elif len(infected_idxes_per_turn) > num_turns:
                infected_idxes_per_turn = infected_idxes_per_turn[:num_turns]

        infection_labels_per_turn = np.zeros((num_nodes, num_turns), dtype=int)
        for turn_idx in range(num_turns):
            for agent_idx in infected_idxes_per_turn[turn_idx]:
                if 0 <= agent_idx < num_nodes:
                    infection_labels_per_turn[agent_idx, turn_idx] = 1

        # Derive final infection labels (keep backward compatibility).
        if num_turns > 0:
            inf_labels = infection_labels_per_turn[:, -1]
        else:
            inf_labels = np.array([1 if i in infected_idxes else 0 for i in range(num_nodes)], dtype=int)

        # Infection labels must not overlap with malicious labels.
        inf_labels = inf_labels * (1 - mal_labels)
        if num_turns > 0:
            infection_labels_per_turn = infection_labels_per_turn * (1 - mal_labels[:, None])
        # edge_index[1] is the destination node index for each edge.
        edge_attr = np.array(communication_embeddings[edge_index[1]], copy=True)
        
        # Node self-reply features: (num_nodes, num_turns, embedding_dim)
        node_self_replies = np.array(communication_embeddings, copy=True)
        
        data = {}
        data["adj_matrix"] = adj_matrix_np
        data["features"] = system_prompts_embedding
        # Two-column labels: [malicious, infected]
        data["labels"] = np.stack([mal_labels, inf_labels], axis=1)
        data["edge_index"] = edge_index
        data["edge_attr"] = edge_attr
        data["node_self_replies"] = node_self_replies
        data["infection_labels_per_turn"] = infection_labels_per_turn
        
        # Extra PI fields.
        if attack_mode == "PI":
            data["question"] = question
            data["correct_answer"] = correct_answer
            data["wrong_answer"] = wrong_answer
        
        dataset.append(data)
        
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)


def get_dataset_path(attack_mode, dataset_name=None, name=''):
    """
    Get dataset path based on attack mode and dataset name.
    
    Args:
        attack_mode: Attack mode
        dataset_name: Dataset name (PI only)
    
    Returns:
        data_dir: Dataset path
    """
    if attack_mode == "PI":
        if dataset_name not in ["mmlu", "csqa", "gsm8k"]:
            raise ValueError(f"PI requires dataset_name in [mmlu, csqa, gsm8k], got: {dataset_name}")
        return f"./output/output_{name}/agent_graph_dataset_{name}/PI/{dataset_name}/train/dataset.json"
    elif attack_mode == "MA":
        return f"./output/output_{name}/agent_graph_dataset_{name}/MA/memory_attack/train/dataset.json"
    elif attack_mode == "TA":
        return f"./output/output_{name}/agent_graph_dataset_{name}/TA/tool_attack/train/dataset.json"
    else:
        raise ValueError(f"Unsupported attack_mode: {attack_mode}")


def get_save_dir(attack_mode, dataset_name=None, name=''):
    """
    Get output directory based on attack mode and dataset name.
    
    Args:
        attack_mode: Attack mode
        dataset_name: Dataset name (PI only)
    
    Returns:
        save_dir: Output directory
    """
    if attack_mode == "PI":
        return f"./output/output_{name}/ModelTrainingSet_{name}/{dataset_name}"
    elif attack_mode == "MA":
        return f"./output/output_{name}/ModelTrainingSet_{name}/memory_attack"
    elif attack_mode == "TA":
        return f"./output/output_{name}/ModelTrainingSet_{name}/tool_attack"
    else:
        raise ValueError(f"Unsupported attack_mode: {attack_mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiments that generate dataset")
    parser.add_argument("--attack_mode", type=str, default="PI", choices=["PI", "MA", "TA"], 
                       help="Attack mode: PI (Prompt Injection), MA (Memory Attack), TA (Tool Attack)")
    parser.add_argument("--dataset", type=str, default="mmlu", choices=["mmlu", "csqa", "gsm8k", 'memory_attack', 'tool_attack'],
                       help="Dataset name (PI only)")
    parser.add_argument("--name", type=str, default="")
    args = parser.parse_args()

    # Resolve dataset path and output dir.
    data_dir = get_dataset_path(args.attack_mode, args.dataset, args.name)
    save_dir = get_save_dir(args.attack_mode, args.dataset, args.name)
    
    if not os.path.exists(save_dir): 
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "dataset.pkl")
    
    # Load embedding model.
    embedding_model_dir = "train/models/sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(embedding_model_dir)
    
    # Load language dataset.
    with open(data_dir, 'r') as file:
        language_dataset = json.load(file)
    
    # Generate training dataset.
    gen_model_training_set(language_dataset, embedding_model, save_path, args.attack_mode)
    
    # Quick sanity check.
    with open(save_path, "rb") as f: 
        loaded_dataset = pickle.load(f)    
    print(f"Generated dataset size: {len(loaded_dataset)}")
    print("First sample:")
    print(loaded_dataset[0])
