import torch
from typing import Literal
import pickle
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.transforms import to_sparse_tensor

class AgentGraphDataset(Dataset): 
    def __init__(self, root, transform=None, phase: Literal["train", "val"]="train"):
        super().__init__()
        with open(root, "rb") as f:
            origin_dataset = pickle.load(f)
        origin_dataset_len = len(origin_dataset)
        if phase == "train": 
            self.dataset = origin_dataset[:int(origin_dataset_len*0.8)]
        elif phase == "val":
            self.dataset = origin_dataset[int(origin_dataset_len*0.8):]
        else:
            raise Exception(f"Unknown phase {phase}")
        # Precompute per-sample turn count T for global bucketing.
        self.turns = []
        for item in self.dataset:
            edge_attr = item["edge_attr"]
            try:
                T = edge_attr.shape[1]
            except Exception:
                # Backward compatibility for list-like structures.
                T = len(edge_attr[0]) if len(edge_attr) > 0 else 0
            self.turns.append(int(T))
    
    def len(self):
        return len(self.dataset)
    
    def get(self, idx):
        origin_data = self.dataset[idx]
        origin_data["adj_matrix"]
        x = torch.tensor(origin_data["features"])
        # labels shape: [num_nodes, 2] -> [malicious, infected]
        y = torch.tensor(origin_data["labels"], dtype=torch.float)
        edge_index = torch.tensor(origin_data["edge_index"])
        edge_attr = torch.tensor(origin_data["edge_attr"])
        
        # Load node self-reply features if present (backward compatible).
        node_self_replies = None
        if "node_self_replies" in origin_data:
            node_self_replies = torch.tensor(origin_data["node_self_replies"])
        
        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = len(x)
        if node_self_replies is not None:
            data.node_self_replies = node_self_replies
        inf_labels_per_turn = origin_data.get("infection_labels_per_turn")
        if inf_labels_per_turn is not None:
            inf_labels_per_turn = torch.tensor(inf_labels_per_turn, dtype=torch.float)
            data.inf_labels_per_turn = inf_labels_per_turn
        
        return data
    