import json
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Experiments that generate dataset")
    parser.add_argument("--attack_mode", type=str, required=True, choices=["PI", "MA", "TA"], 
                       help="Attack mode: PI (Prompt Injection), MA (Memory Attack), TA (Tool Attack)")
    parser.add_argument("--dataset", type=str, default="mmlu", choices=["mmlu", "csqa", "gsm8k", "memory_attack", "tool_attack"],
                       help="Dataset type for PI attack mode (required for PI, ignored for MA/TA)")
    parser.add_argument("--phase", type=str, default="train", choices=["train", "test", "validation"],
                       help="Phase: train or test")
    parser.add_argument("--root", type=str, default=None,
                       help="Custom root directory (optional, will use default if not provided)")
    
    args = parser.parse_args()

    # Set default root paths based on attack_mode
    if args.root is None:
        if args.attack_mode == "PI":
            if args.dataset == "mmlu":
                args.root = "./agent_graph_dataset/PI/mmlu"
            elif args.dataset == "csqa":
                args.root = "./agent_graph_dataset/PI/csqa"
            elif args.dataset == "gsm8k":
                args.root = "./agent_graph_dataset/PI/gsm8k"
            else:
                raise Exception(f"Unknown dataset {args.dataset} for PI attack mode")
        elif args.attack_mode == "MA":
            args.root = "./agent_graph_dataset/MA/memory_attack"
        elif args.attack_mode == "TA":
            args.root = "./agent_graph_dataset/TA/tool_attack"
        else:
            raise Exception(f"Unknown attack mode {args.attack_mode}")

    root = os.path.join(args.root, args.phase)
    
    # Check if directory exists
    if not os.path.exists(root):
        raise Exception(f"Directory {root} does not exist")

    files = os.listdir(root)

    datafiles = []
    for file in files: 
        data_file = os.path.join(root, file)
        datafiles.append(data_file)

    dataset = []
    for datafile in datafiles: 
        with open(datafile, "r") as f: 
            data = json.load(f)
        dataset += data

    save_file = os.path.join(root, "dataset.json")
    with open(save_file, "w") as f:
        json.dump(dataset, f, indent=None)
    
    print(f"Successfully merged {len(dataset)} data points from {len(datafiles)} files")
    print(f"Saved merged dataset to: {save_file}")


if __name__ == "__main__":
    main()
