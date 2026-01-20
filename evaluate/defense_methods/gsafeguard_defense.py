"""
GSAFeguard defense implementation.
This is a simplified version that only detects malicious nodes.
"""

import asyncio
import numpy as np
import torch
import copy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from torch_scatter import scatter_mean
from train.models.defender.model import MyGAT
from MAS.agents import AgentGraphWithDefense


def get_sentence_embedding(sentence):
    """Get a sentence embedding."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("./train/models/sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(sentence)
    return embeddings


def response2embeddings(responses): 
    """Convert responses into embedding vectors."""
    embeddings = [None for _ in range(len(responses))]
    for agent_idx, agent_response in responses: 
        # Extract response text.
        if isinstance(agent_response, dict):
            response_text = agent_response.get('answer', '') or str(agent_response)
        else:
            response_text = str(agent_response)
        embeddings[agent_idx] = get_sentence_embedding(response_text)
    
    embeddings = np.array(embeddings)
    return embeddings


def embeddings2graph_gsafeguard(embeddings, adj_matrix):
    """Graph construction for the simplified GSAFeguard variant."""
    edge_index = torch.tensor(np.array(adj_matrix.nonzero()))
    edge_attr = torch.tensor(np.array(embeddings))[:, edge_index[1]]
    x = edge_attr[0, :]
    x = scatter_mean(x, edge_index[1], dim=0, dim_size=len(embeddings[0]))
    edge_attr = edge_attr.transpose(0, 1)
    return x, edge_index, edge_attr


async def defense_communication_gsafeguard(
    ag: AgentGraphWithDefense, 
    gnn: MyGAT, 
    input_data, 
    adj_m: np.ndarray, 
    num_dialogue_turns: int, 
    attack_mode: str = "PI"
):
    """
    GSAFeguard defense communication (simplified: malicious-node detection only).
    
    Args:
        ag: AgentGraphWithDefense instance.
        gnn: GNN model.
        input_data: Input data.
        adj_m: Adjacency matrix.
        num_dialogue_turns: Number of dialogue turns.
        attack_mode: Attack mode (PI/MA/TA).
    
    Returns:
        communication_data: List of turns, each containing all agents' responses.
        In MA mode, also returns identified_attackers.
    """
    communication_data = []
    response_embeddings = []
    identified_attackers = []
    
    # First turn.
    if attack_mode == "PI":
        initial_responses = await ag.afirst_generate(input_data)
    elif attack_mode == "MA":
        initial_responses = await ag.afirst_generate(input_data[0], input_data[1])
    elif attack_mode == "TA":
        initial_responses = await ag.afirst_generate(input_data)
    else:
        raise ValueError(f"Unsupported attack mode: {attack_mode}")
    
    embeddings = response2embeddings(initial_responses)
    response_embeddings.append(embeddings)
    x, edge_index, edge_attr = embeddings2graph_gsafeguard(response_embeddings, adj_m)
    predicts = torch.sigmoid(gnn(x, edge_index, edge_attr).squeeze(-1)) >= 0.5
    
    # Set attacker roles.
    for idx, predict in enumerate(predicts): 
        if predict == 1: 
            ag.agents[idx].set_role("attacker")
    
    communication_data.append(initial_responses)
    
    # In MA mode, record identified attackers.
    if attack_mode == "MA":
        identified = []
        for idx, predict in enumerate(predicts):
            if predict == 1 and idx not in identified:
                identified.append(idx)
        identified_attackers.append(copy.deepcopy(identified))

    # Subsequent turns.
    for _ in range(num_dialogue_turns): 
        responses = await ag.are_generate()
        embeddings = response2embeddings(responses)
        response_embeddings.append(embeddings)
        x, edge_index, edge_attr = embeddings2graph_gsafeguard(response_embeddings, adj_m)
        predicts = torch.sigmoid(gnn(x, edge_index, edge_attr).squeeze(-1)) >= 0.5
        
        # Update attacker roles.
        for idx, predict in enumerate(predicts): 
            if predict == 1: 
                ag.agents[idx].set_role("attacker")
        
        communication_data.append(responses)
        
        # In MA mode, update identified attackers.
        if attack_mode == "MA":
            for idx, predict in enumerate(predicts):
                if predict == 1 and idx not in identified:
                    identified.append(idx)
            identified_attackers.append(copy.deepcopy(identified))

    if attack_mode == "MA":
        return communication_data, identified_attackers
    else:
        return communication_data

