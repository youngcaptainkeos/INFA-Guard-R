from train.models.defender.gat_with_attr_conv import GATwithEdgeConv
import torch.nn as nn
import torch.nn.functional as F
import torch
from einops import rearrange


class DiaglogueEmbeddingProcessModules(nn.Module):
    def __init__(self, aggr_type, edge_dim, max_turns=3, add_time_emb=False):
        super().__init__()
        self.aggr_type = aggr_type
        self.add_time_emb = add_time_emb 

    def forward(self, diag_emb: torch.Tensor): 
        # edge_attr is shaped as (num_edges, num_turns, embedding_dim)
        # node_self_replies is shaped as (num_nodes, num_turns, embedding_dim)
        # diag_emb: [num_edges, T, D]
        # print(diag_emb.size())
        if self.aggr_type == "last":
            base = diag_emb[:, -1, :]
        elif self.aggr_type == "mean":
            base = diag_emb.mean(dim=1)
        else: 
            raise Exception("Not a correct method of aggregation!")

        # residual: last - last last (if T>=2), else zeros
        if diag_emb.size(1) >= 2:
            res = diag_emb[:, -1, :] - diag_emb[:, -2, :]
        else:
            res = torch.zeros_like(base)

        # trend: mean over time
        trend = diag_emb.mean(dim=1)
        
        return base, res, trend


class MyGAT(nn.Module): 
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, concat=True, edge_dim=None, num_layers=2, dropout=0.2, residual=False, aggr_type="mean", add_time_emb=False, turn_ranges=[(1, 1), (2, 2), (3, 3), (4,5)], guard="ours"):
        """
        turn_ranges: [(min_turns, max_turns), ...] defines branches for different turn ranges.
        Default: [(1,1), (2,3), (4,5)] means 3 branches: 1 turn, 2-3 turns, 4-5 turns.
        guard: "gsafeguard" or "ours" controls which model structure to use.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.guard = guard
        self.turn_ranges = turn_ranges
        self.num_branches = len(turn_ranges) if guard == "ours" else 0
    
        self.heads = heads
        self.head_channels = hidden_channels // heads
        self.hidden_channels = self.head_channels * heads

        max_turns, edge_dim = self.edge_dim
        self.edge_feature_dim = edge_dim
        
        if guard == "gsafeguard":
            # GSAFeguard variant: simple model structure.
            self.convs = nn.ModuleList()
            conv1 = GATwithEdgeConv(in_channels, self.head_channels, heads=heads, concat=concat, edge_dim=edge_dim, residual=residual)
            self.convs.append(conv1)
            for i in range(num_layers-1):
                conv_i = GATwithEdgeConv(self.hidden_channels, self.head_channels, heads=heads, concat=concat, edge_dim=hidden_channels, residual=residual)
                self.convs.append(conv_i)
            
            # GSAFeguard embedding processor (returns a single embedding).
            class SimpleDiaglogueEmbeddingProcessModules(nn.Module):
                def __init__(self, aggr_type, edge_dim, max_turns=3, add_time_emb=False):
                    super().__init__()
                    self.aggr_type = aggr_type
                    self.add_time_emb = add_time_emb 

                def forward(self, diag_emb: torch.Tensor): 
                    # print(diag_emb.size())
                    if self.aggr_type == "last":
                        emb = diag_emb[:, -1, :]
                    elif self.aggr_type == "mean":
                        emb = diag_emb.mean(dim=1)
                    else: 
                        raise Exception("Not a correct method of aggregation!")
                    return emb
            
            self.diag_emb_proc = SimpleDiaglogueEmbeddingProcessModules(aggr_type, edge_dim, max_turns, add_time_emb)
            self.out = nn.Linear(self.hidden_channels, out_channels)
        else:
            # Our variant: branched structure.
            # Shared backbone layers (shared by all branches).
            self.shared_convs = nn.ModuleList()
            conv1 = GATwithEdgeConv(in_channels, self.head_channels, heads=heads, concat=concat, edge_dim=edge_dim, residual=residual)
            self.shared_convs.append(conv1)
            
            # Progressive branch layers: each branch corresponds to a turn range and may have extra depth.
            self.branch_convs = nn.ModuleList()
            for branch_idx, (min_t, max_t) in enumerate(turn_ranges):
                branch_layers = nn.ModuleList()
                # Branch depth increases with turn range (cap at 3 extra layers).
                branch_depth = min(branch_idx + 1, 3)
                for i in range(branch_depth):
                    conv_i = GATwithEdgeConv(self.hidden_channels, self.head_channels, heads=heads, concat=concat, edge_dim=hidden_channels, residual=residual)
                    branch_layers.append(conv_i)
                self.branch_convs.append(branch_layers)
            
            self.diag_emb_proc = DiaglogueEmbeddingProcessModules(aggr_type, edge_dim, max_turns, add_time_emb)
            
            # Node self-reply feature processor.
            self.node_self_reply_proc = DiaglogueEmbeddingProcessModules(aggr_type, edge_dim, max_turns, add_time_emb)
            # If using self-replies, we expand the fused feature dimension.
            self.use_self_replies = True  # Can be controlled via a parameter.
            
            # Concatenated feature dimension (including self-reply base/res/trend).
            if self.use_self_replies:
                concat_dim = self.hidden_channels + 2 * self.edge_feature_dim + 3 * self.edge_feature_dim
            else:
                concat_dim = self.hidden_channels + 2 * self.edge_feature_dim
            
            # Input projection: project fused features back to in_channels.
            # x is aggregated from edge_attr and typically has dimension in_channels (embedding_dim).
            if self.use_self_replies:
                input_fusion_dim = in_channels + 5 * self.edge_feature_dim
            else:
                input_fusion_dim = in_channels + 2 * self.edge_feature_dim
            self.input_proj = nn.Linear(input_fusion_dim, in_channels)
            
            # Per-branch MLPs and classification heads.
            shared_mlp_dim = self.hidden_channels
            self.branch_mlps = nn.ModuleList()
            self.branch_inf_mlps = nn.ModuleList()
            self.branch_heads_mal = nn.ModuleList()
            self.branch_heads_inf = nn.ModuleList()
            
            for _ in range(self.num_branches):
                # Shared MLP layers.
                shared_mlp = nn.Sequential(
                    nn.Linear(concat_dim, shared_mlp_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(shared_mlp_dim, shared_mlp_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                self.branch_mlps.append(shared_mlp)
                
                # Infection-task-specific MLP.
                inf_mlp = nn.Sequential(
                    nn.Linear(concat_dim, shared_mlp_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(shared_mlp_dim, shared_mlp_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                self.branch_inf_mlps.append(inf_mlp)
                
                # Heads.
                self.branch_heads_mal.append(nn.Linear(shared_mlp_dim, 1))
                self.branch_heads_inf.append(nn.Linear(shared_mlp_dim, 1))
    
    def _select_branch(self, num_turns):
        """Select the branch index based on the number of turns."""
        for branch_idx, (min_t, max_t) in enumerate(self.turn_ranges):
            if min_t <= num_turns <= max_t:
                return branch_idx
        # If out of range, use the last branch (handle long sequences).
        return len(self.turn_ranges) - 1
    
    def _forward_branch(self, x, branch_idx, edge_index, edge_attr):
        """Forward through the specified branch."""
        branch_layers = self.branch_convs[branch_idx]
        for conv in branch_layers:
            x, edge_attr = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def _forward_heads(self, x, branch_idx, res_edge_attr, trend_edge_attr, edge_index,
                       node_self_reply_base=None, node_self_reply_res=None, node_self_reply_trend=None):
        """Forward through the heads for the specified branch."""
        from torch_scatter import scatter_mean
        num_nodes = x.size(0)
        node_res = scatter_mean(res_edge_attr, edge_index[0], dim=0, dim_size=num_nodes)
        node_trend = scatter_mean(trend_edge_attr, edge_index[0], dim=0, dim_size=num_nodes)
        
        # Concatenate features.
        if node_self_reply_base is not None and self.use_self_replies:
            concat_features = torch.cat([x, node_res, node_trend, 
                                        node_self_reply_base, node_self_reply_res, node_self_reply_trend], dim=-1)
        else:
            concat_features = torch.cat([x, node_res, node_trend], dim=-1)
        
        # Branch MLPs.
        shared_features = self.branch_mlps[branch_idx](concat_features)
        inf_features = self.branch_inf_mlps[branch_idx](concat_features)
        
        # Heads.
        logit_mal = self.branch_heads_mal[branch_idx](shared_features)
        logit_inf = self.branch_heads_inf[branch_idx](inf_features)
        
        return torch.cat([logit_mal, logit_inf], dim=-1)
    
    def _forward_all_branches(self, x, edge_index, edge_attr, res_edge_attr, trend_edge_attr,
                              node_self_reply_base=None, node_self_reply_res=None, node_self_reply_trend=None):
        """Aggregate outputs from all branches at inference time (simple average)."""
        outputs_list = []
        for branch_idx in range(self.num_branches):
            branch_x = self._forward_branch(x, branch_idx, edge_index, edge_attr)
            branch_output = self._forward_heads(branch_x, branch_idx, res_edge_attr, trend_edge_attr, edge_index,
                                               node_self_reply_base, node_self_reply_res, node_self_reply_trend)
            outputs_list.append(branch_output)
        
        # Simple average (could be replaced by weighted average).
        output = torch.stack(outputs_list, dim=0).mean(dim=0)
        return output
    
    def forward(self, x, edge_index, edge_attr, num_turns=None, node_self_replies=None):
        if self.guard == "gsafeguard":
            # Forward for the GSAFeguard variant.
            edge_attr = self.diag_emb_proc(edge_attr)
            for i in range(self.num_layers): 
                x, edge_attr = self.convs[i](x, edge_index, edge_attr=edge_attr)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.out(x)
            return x
        
        # Forward for our variant.
        """
        num_turns: number of dialogue turns for the current batch (used for branch selection).
        If None, aggregate all branches (inference mode).
        node_self_replies: per-node self-reply features shaped (num_nodes, num_turns, embedding_dim).
        """
        # Extract base/res/trend from temporal edge features.
        base_edge_attr, res_edge_attr, trend_edge_attr = self.diag_emb_proc(edge_attr)

        # Process node self-replies: also extract base/res/trend (consistent with edge_attr).
        node_self_reply_base = None
        node_self_reply_res = None
        node_self_reply_trend = None
        if node_self_replies is not None and self.use_self_replies:
            base_self_reply, res_self_reply, trend_self_reply = self.node_self_reply_proc(node_self_replies)
            node_self_reply_base = base_self_reply
            node_self_reply_res = res_self_reply
            node_self_reply_trend = trend_self_reply

        # Aggregate edge res/trend to node features.
        from torch_scatter import scatter_mean
        num_nodes = x.size(0)
        node_edge_res = scatter_mean(res_edge_attr, edge_index[0], dim=0, dim_size=num_nodes)
        node_edge_trend = scatter_mean(trend_edge_attr, edge_index[0], dim=0, dim_size=num_nodes)
        
        # Fuse features before the shared backbone.
        # x is aggregated from edge_attr; optionally concat node_self_reply_base.
        if node_self_reply_base is not None and self.use_self_replies:
            # x + node_self_reply_base + edge_res + edge_trend + self_reply_res + self_reply_trend
            x = torch.cat([x, node_self_reply_base, node_edge_res, node_edge_trend, node_self_reply_res, node_self_reply_trend], dim=-1)
        else:
            # Fuse only edge res/trend (x is aggregated from edge_attr).
            x = torch.cat([x, node_edge_res, node_edge_trend], dim=-1)
            # If use_self_replies=True but node_self_replies is missing, pad zeros to match input_proj dims.
            if self.use_self_replies:
                # Add zero vectors for node_self_reply_base/self_reply_res/self_reply_trend.
                zero_base = torch.zeros(num_nodes, self.edge_feature_dim, device=x.device, dtype=x.dtype)
                zero_res = torch.zeros_like(node_edge_res)
                zero_trend = torch.zeros_like(node_edge_trend)
                x = torch.cat([x, zero_base, zero_res, zero_trend], dim=-1)
        # Project back to in_channels for the shared backbone.
        x = self.input_proj(x)

        # Shared backbone.
        edge_attr = base_edge_attr
        for conv in self.shared_convs:
            x, edge_attr = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        shared_features = x
        
        # Choose a branch by num_turns, or aggregate all branches.
        if num_turns is not None:
            # Training/eval with known num_turns: select the corresponding branch.
            branch_idx = self._select_branch(num_turns)
            x = self._forward_branch(shared_features, branch_idx, edge_index, edge_attr)
            output = self._forward_heads(x, branch_idx, res_edge_attr, trend_edge_attr, edge_index,
                                       node_self_reply_base, node_self_reply_res, node_self_reply_trend)
        else:
            # Inference with unknown num_turns: aggregate all branches.
            output = self._forward_all_branches(shared_features, edge_index, edge_attr, res_edge_attr, trend_edge_attr,
                                              node_self_reply_base, node_self_reply_res, node_self_reply_trend)
        
        return output


if __name__ == "__main__":
    x = torch.randn((2, 100))
    edge_index = torch.tensor([[0], [1]])
    model = MyGAT(100, 200, 1, heads=2, concat=True)
    y = model(x, edge_index)
    print(y)
