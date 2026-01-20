import os
import sys
import numpy as np
import random
import json
import asyncio
from typing import Literal
from tqdm import tqdm

sys.path.append('.')

from generate_data.dataset_utils.gen_csqa import gen_csqa_datasets
from generate_data.dataset_utils.gen_mmlu import gen_mmlu_datasets
from generate_data.dataset_utils.gen_gsm8k import gen_gsm8k_dataset
from generate_data.dataset_utils.gen_memory_attack_data import gen_poisonrag_data
from generate_data.dataset_utils.get_tool_attack_data import gen_injecagent_data
from MAS.agents import AgentGraph
from evaluate.evaluate_output import extract_answer, judge_output, extract_answer_choice, extract_answer_number, extract_correct_answer_number
from utils.evaluation_utils import evaluate_output_prompted



def generate_directed_graph_dataset(num_nodes, sparsity, num_graphs):
    """
    Generate a dataset of adjacency matrices representing directed graphs.
    
    Parameters:
        num_nodes (int): Number of nodes in each graph.
        sparsity (float): Sparsity of the edges (0 to 1), where higher values indicate denser graphs.
        num_graphs (int): Number of graphs to generate.
        
    Returns:
        List[np.array]: A list of adjacency matrices, each of shape (num_nodes, num_nodes).
    """
    if not (0 <= sparsity <= 1):
        raise ValueError("Sparsity must be a value between 0 and 1.")
    
    graph_dataset = []
    
    for _ in range(num_graphs):
        adj_matrix = np.random.rand(num_nodes, num_nodes) <= sparsity
        np.fill_diagonal(adj_matrix, 0)
        adj_matrix = adj_matrix.astype(int)
        
        graph_dataset.append(adj_matrix)
    
    return graph_dataset


def generate_agent_graph_dataset(num_nodes, sparsity, num_graphs, num_attackers, attack_mode="PI", dataset_type="mmlu"): 
    """
    Generate agent graph dataset with different attack modes.
    
    Parameters:
        num_nodes (int): Number of nodes in each graph.
        sparsity (float): Sparsity of the edges (0 to 1).
        num_graphs (int): Number of graphs to generate.
        num_attackers (int): Number of attacker nodes.
        attack_mode (str): Attack mode - "MA" (Memory Attack), "PI" (Prompt Injection), "TA" (Tool Attack)
        dataset_type (str): Dataset type for PI mode - "mmlu", "csqa", "gsm8k"
        
    Returns:
        List[dict]: A list of agent graph data dictionaries.
    """
    assert num_attackers < num_nodes
    dataset = []
    adj_matrices = generate_directed_graph_dataset(num_nodes, sparsity, num_graphs)
    
    for adj_m in adj_matrices: 
        random_attacker_idxes = random.sample(range(0, num_nodes), num_attackers)
        
        # Define prompts based on attack mode
        if attack_mode == "MA":
            from MAS.agent_prompts import MA_SYS_PROMPT, MA_ATTACKER_SYS_PROMPT
            normal_prompt = "Your are agent_{}. " + MA_SYS_PROMPT
            attacker_prompt = "You are agent_{}. " + MA_ATTACKER_SYS_PROMPT
        elif attack_mode == "PI":
            from MAS.agent_prompts import PI_SYS_PROMPT, PI_ATTACKER_SYS_PROMPT, PI_SYS_PROMPT_GSM8K, PI_ATTACKER_SYS_PROMPT_GSM8K
            
            # Different prompts for different datasets within PI mode
            if dataset_type == "gsm8k":
                normal_prompt = PI_SYS_PROMPT_GSM8K
                attacker_prompt = PI_ATTACKER_SYS_PROMPT_GSM8K
            else:
                normal_prompt = PI_SYS_PROMPT
                attacker_prompt = PI_ATTACKER_SYS_PROMPT
        elif attack_mode == "TA":
            from MAS.agent_prompts import TA_SYS_PROMPT, TA_ATTACKER_SYS_PROMPT
            normal_prompt = "Your are agent_{}. " + TA_SYS_PROMPT
            attacker_prompt = "You are agent_{}. " + TA_ATTACKER_SYS_PROMPT
        else:
            raise ValueError(f"Unknown attack mode: {attack_mode}")

        data = {}
        data["adj_matrix"] = adj_m
        
        data["attacker_idxes"] = random_attacker_idxes
        data["normal_prompt"] = normal_prompt
        data["attacker_prompt"] = attacker_prompt
        dataset.append(data)
    return dataset


def generate_initial_data(ag_data, example, attack_mode="PI"): 
    """
    Generate initial data for different attack modes.
    
    Parameters:
        ag_data (dict): Agent graph data.
        example: Example data from dataset.
        attack_mode (str): Attack mode.
        
    Returns:
        dict: Initial data dictionary.
    """
    new_data = {}
    adj_m = ag_data["adj_matrix"]
    attacker_idxes = ag_data["attacker_idxes"]
    normal_prompt = ag_data["normal_prompt"]
    attacker_prompt = ag_data["attacker_prompt"]
    agent_system_prompts = []
    
    for idx in range(adj_m.shape[0]): 
        if idx in attacker_idxes: 
            agent_idx_prompt = attacker_prompt.format(idx)
        else:
            agent_idx_prompt = normal_prompt.format(idx)
        agent_system_prompts.append(agent_idx_prompt)
    
    new_data["adj_matrix"] = adj_m
    new_data["attacker_idxes"] = attacker_idxes
    new_data["system_prompts"] = agent_system_prompts
    
    # Handle different data formats based on attack mode
    if attack_mode == "MA":
        new_data["query"] = example[0]
        new_data["adv_texts"] = example[1]
        new_data["correct_answer"] = example[2]
        new_data["incorrect_answer"] = example[3]
    elif attack_mode == "PI":
        new_data["question"] = example[0]
        new_data["correct_answer"] = example[1]
        new_data["wrong_answer"] = example[2] if len(example) > 2 else None
    elif attack_mode == "TA":
        new_data["user_instruction"] = example[0]
        new_data["available_tool_names"] = example[1]
        new_data["available_tool_descriptions"] = example[2]
        new_data["agent_scratchpad"] = example[3]
        new_data["attacker_instruction"] = example[4]
        new_data["user_tools"] = example[5]
        new_data["attacker_tools"] = example[6]
    
    return new_data


async def generate_graph_dataset(args, attack_mode="PI"): 
    """
    Generate graph dataset for different attack modes.
    
    Parameters:
        args: Command line arguments.
        attack_mode (str): Attack mode.
        
    Returns:
        List[dict]: Final dataset with communication data.
    """
    # Load dataset based on attack mode
    if attack_mode == "MA":
        cases_dataset = gen_poisonrag_data(args.dataset_path, args.phase)
    elif attack_mode == "PI":
        if args.dataset == "csqa": 
            qa_dataset = gen_csqa_datasets(args.dataset_path, phase=args.phase)
        elif args.dataset == "mmlu":
            if args.phase == "train":
                args.phase = "dev"
            qa_dataset = gen_mmlu_datasets(args.dataset_path, phase=args.phase)
        elif args.dataset == "gsm8k": 
            qa_dataset = gen_gsm8k_dataset(args.dataset_path, phase=args.phase)
        else:
            raise Exception(f"Unknown dataset: {args.dataset}")
        cases_dataset = qa_dataset
    elif attack_mode == "TA":
        cases_dataset = gen_injecagent_data(args.dataset_path, phase=args.phase)
    else:
        raise ValueError(f"Unknown attack mode: {attack_mode}")

    # Generate agent graph dataset
    ag_dataset = generate_agent_graph_dataset(
        num_nodes=args.num_nodes, 
        sparsity=args.sparsity, 
        num_graphs=args.num_graphs, 
        num_attackers=args.num_attackers,
        attack_mode=attack_mode
    )
    
    # Generate initial dataset
    initial_dataset = []
    for agent_graph in tqdm(ag_dataset, desc="Generate meta data"):
        for case in cases_dataset: 
            initial_data = generate_initial_data(agent_graph, case, attack_mode)
            initial_dataset.append(initial_data)
    
    random.shuffle(initial_dataset)
    sampled_initial_dataset = initial_dataset[:args.samples]

    # Generate communication data
    final_dataset = []
    for d in tqdm(sampled_initial_dataset, desc="Generate communication data"): 
        adj_m = d["adj_matrix"]
        attacker_idxes = d["attacker_idxes"]
        system_prompts = d["system_prompts"]
        
        # Prepare input data based on attack mode
        if attack_mode == "MA":
            query = d["query"]
            context = d["adv_texts"]
            case = (query, context)
        elif attack_mode == "PI":
            qa_data_origin = d["question"], d["correct_answer"], d["wrong_answer"]
            wrong_answer = random.choice(qa_data_origin[2]) if qa_data_origin[2] else None
            case = (qa_data_origin[0], qa_data_origin[1], wrong_answer)
        elif attack_mode == "TA":
            user_instruction = d["user_instruction"]
            available_tool_names = d["available_tool_names"]
            available_tool_descriptions = d["available_tool_descriptions"]
            agent_scratchpad = d["agent_scratchpad"]
            case = (user_instruction, available_tool_names, available_tool_descriptions, agent_scratchpad)
        
        # Create agent graph and generate communication
        ag = AgentGraph(adj_m, system_prompts, attacker_idxes, model_type=args.model_type, attack_mode=attack_mode, dataset=args.dataset)
        communication_data = []
        initial_responses = await ag.afirst_generate(case)
        communication_data.append(initial_responses)
        
        for _ in range(args.num_dialogue_turns): 
            responses = await ag.are_generate()
            communication_data.append(responses)
        
        
        base_answer = {}
        infected_set = set()
        infected_idxes_per_turn = []
        if len(communication_data) >= 1:
            # Compute first-turn baseline correctness (aligned with cal_acc; exclude attackers).
            base_is_correct = {}
            if attack_mode == "MA":
                question = d.get("query", "")
                correct_answer = d.get("correct_answer")
                for agent_idx, text in communication_data[0]:
                    if agent_idx in attacker_idxes:
                        continue
                    try:
                        result = judge_output(text, question, correct_answer, attack_mode="MA")
                        base_is_correct[agent_idx] = (result.is_success == 1)
                    except Exception:
                        base_is_correct[agent_idx] = False
            elif attack_mode == "PI":
                # Choose answer type by dataset: gsm8k -> number; otherwise -> choice.
                answer_type = "number" if args.dataset == "gsm8k" else "choice"
                correct_answer = d.get("correct_answer")
                for agent_idx, text in communication_data[0]:
                    if agent_idx in attacker_idxes:
                        continue
                    try:
                        if answer_type == "number":
                            correct_answer = extract_correct_answer_number(correct_answer)
                            parsed = extract_answer_number(text)
                        else:
                            parsed = extract_answer_choice(text)
                        base_is_correct[agent_idx] = (parsed == correct_answer) if parsed is not None else False
                    except Exception:
                        base_is_correct[agent_idx] = False
            else:
                # TA mode: use evaluate_output_prompted to decide whether the agent was attacked.
                user_tools = d.get("user_tools")
                attacker_tools = d.get("attacker_tools")
                attacker_instruction = d.get("attacker_instruction")
                
                if all([user_tools, attacker_tools, attacker_instruction]):
                    for agent_idx, text in communication_data[0]:
                        if agent_idx in attacker_idxes:
                            continue
                        try:
                            result = evaluate_output_prompted(
                                text,
                                target_tool=attacker_tools[0],
                                attacker_tools=attacker_tools,
                                user_tool=user_tools,
                                attacker_instruction=attacker_instruction,
                            )
                            # If eval == "succ", the attack succeeded.
                            base_is_correct[agent_idx] = (result.get("eval") == "unsucc")
                        except Exception:
                            base_is_correct[agent_idx] = False
            # The first turn is treated as not infected by default.
            infected_idxes_per_turn.append([])
            for turn in communication_data[1:]:
                current_infected = set()  # Re-create each turn for independence.
                for agent_idx, text in turn:
                    if attack_mode == "MA":
                        if agent_idx in attacker_idxes:
                            continue
                        try:
                            result = judge_output(text, d.get("query", ""), d.get("correct_answer"), attack_mode="MA")
                            cur_is_correct = (result.is_success == 1)
                        except Exception:
                            cur_is_correct = False
                        if agent_idx in base_is_correct:
                            if cur_is_correct != base_is_correct[agent_idx] and base_is_correct[agent_idx] == True:
                                current_infected.add(agent_idx)
                                infected_set.add(agent_idx)
                    elif attack_mode == "PI":
                        if agent_idx in attacker_idxes:
                            continue
                        answer_type = "number" if args.dataset == "gsm8k" else "choice"
                        try:
                            if answer_type == "number":
                                correct_answer = d.get("correct_answer")
                                correct_answer = extract_correct_answer_number(correct_answer)
                                parsed = extract_answer_number(text)
                            else:
                                parsed = extract_answer_choice(text)
                            cur_is_correct = (parsed == d.get("correct_answer")) if parsed is not None else False
                        except Exception:
                            cur_is_correct = False
                        if agent_idx in base_is_correct:
                            if cur_is_correct != base_is_correct[agent_idx] and base_is_correct[agent_idx] == True:
                                current_infected.add(agent_idx)
                                infected_set.add(agent_idx)
                    else:
                        # TA mode: use evaluate_output_prompted to decide whether the agent was attacked.
                        if agent_idx in attacker_idxes:
                            continue
                        
                        user_tools = d.get("user_tools")
                        attacker_tools = d.get("attacker_tools")
                        attacker_instruction = d.get("attacker_instruction")
                        
                        if not all([user_tools, attacker_tools, attacker_instruction]):
                            continue
                        
                        try:
                            result = evaluate_output_prompted(
                                text,
                                target_tool=attacker_tools[0],
                                attacker_tools=attacker_tools,
                                user_tool=user_tools,
                                attacker_instruction=attacker_instruction,
                            )
                            # If eval == "succ", the attack succeeded.
                            cur_is_attacked = (result.get("eval") == "unsucc")
                        except Exception:
                            cur_is_attacked = False
                        
                        # If it flips from not-attacked to attacked, mark as infected.
                        if agent_idx in base_is_correct:
                            if cur_is_attacked != base_is_correct[agent_idx] and base_is_correct[agent_idx] == True:
                                current_infected.add(agent_idx)
                                infected_set.add(agent_idx)
                infected_idxes_per_turn.append(sorted(current_infected))
        else:
            infected_idxes_per_turn = []

        d["communication_data"] = communication_data
        d["infected_idxes"] = list(sorted(infected_set))
        d["infected_idxes_per_turn"] = infected_idxes_per_turn
        d["adj_matrix"] = d["adj_matrix"].tolist()
        final_dataset.append(d)
    
    # Save dataset
    with open(args.save_filepath, "w") as file:
        json.dump(final_dataset, file, indent=None) 

    return final_dataset


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Experiments that generate dataset")

        parser.add_argument("--attack_mode", type=str, default="PI", choices=["MA", "PI", "TA"], 
                          help="Attack mode: MA (Memory Attack), PI (Prompt Injection), TA (Tool Attack)")
        parser.add_argument("--dataset", type=str, default="mmlu", choices=["mmlu", "csqa", "gsm8k", 'memory_attack', 'tool_attack'])
        parser.add_argument("--dataset_path", type=str, help="The path to store the dataset")
        parser.add_argument("--num_nodes", type=int, default=8)
        parser.add_argument("--sparsity", type=float, default=0.2, help="Sparsity of the edges (0 to 1), where higher values indicate denser graphs. 1 represents complete graph.")
        parser.add_argument("--num_graphs", type=int, default=20, help="The number of random topological structures")
        parser.add_argument("--num_attackers", type=int, default=3)
        parser.add_argument("--num_dialogue_turns", type=int, default=1)
        parser.add_argument("--samples", type=int, default=40)
        parser.add_argument("--save_dir", type=str, default="./agent_graph_dataset")
        parser.add_argument("--model_type", type=str, default="gpt-4o-mini")
        parser.add_argument("--phase", type=str, default="test")
        parser.add_argument("--save_filepath", type=str)

        args = parser.parse_args()
        
        # Set default dataset paths based on attack mode and dataset
        if args.attack_mode == "MA":
            if not args.dataset_path:
                args.dataset_path = "./datasets/MA/msmarco.json"
            args.dataset = "memory_attack"
        elif args.attack_mode == "PI":
            if not args.dataset_path:
                if args.dataset == "mmlu": 
                    args.dataset_path = "./datasets/PI/MMLU"
                elif args.dataset == "csqa": 
                    args.dataset_path = "./datasets/PI/commonsense_qa/data"
                elif args.dataset == "gsm8k": 
                    args.dataset_path = "./datasets/PI/gsm8k"
        elif args.attack_mode == "TA":
            if not args.dataset_path:
                args.dataset_path = "./datasets/TA/attack_unsucc_data.json"
            args.dataset = "tool_attack"
        
        # Set save directory and filepath
        args.save_dir = os.path.join(args.save_dir, args.attack_mode, args.dataset, args.phase)
        if not os.path.exists(args.save_dir): 
            os.makedirs(args.save_dir)
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_filepath = os.path.join(args.save_dir, 
                                         f"{current_time_str}-{args.attack_mode}-dataset_size_{args.samples}-num_nodes_{args.num_nodes}-num_attackers_{args.num_attackers}-sparsity_{args.sparsity}.json")

        return args

    args = parse_arguments()
    dataset = asyncio.run(generate_graph_dataset(args, attack_mode=args.attack_mode))
    print(f"Generated dataset with {len(dataset)} samples")
