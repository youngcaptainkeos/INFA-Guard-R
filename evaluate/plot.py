import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse

def read_json_files_from_directory(root_dir):
    """Read all JSON files under a root directory (recursive)."""
    pattern = os.path.join(root_dir, "**", "*.json")
    json_files = glob.glob(pattern, recursive=True)

    # Exclude summary files containing "_all" in the filename.
    json_files = [f for f in json_files if "_all" not in os.path.basename(f)]
        
    data = {}
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data[file_path] = json.load(f)
            print(f"Loaded: {file_path}")
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
    
    return data

def extract_file_label(file_path, data):
    filename = os.path.basename(file_path)
    
    # Prefer metadata in-file; fall back to parsing from filename.
    attack_mode = data.get('attack_mode', 'unknown')
    defense_type = data.get('defense_type', 'unknown')
    
    # If missing, parse from filename.
    if attack_mode == 'unknown':
        if 'PI' in filename:
            attack_mode = 'PI'
        elif 'MA' in filename:
            attack_mode = 'MA'
        elif 'TA' in filename:
            attack_mode = 'TA'
    
    if defense_type == 'unknown':
        if 'no_defense' in filename:
            defense_type = 'no_defense'
        elif 'defense' in filename:
            defense_type = 'defense'
    
    # Build a display label.
    label_parts = []
    if attack_mode != 'unknown':
        label_parts.append(attack_mode)
    if defense_type != 'unknown':
        label_parts.append(defense_type)
    
    if label_parts:
        label = ' '.join(label_parts)
    else:
        label = os.path.splitext(filename)[0]
    return label

# Compute per-position mean/std.
def calculate_stats(data_dict):
    stats = {}
    
    for file_path, data in data_dict.items():
        if "_all" in file_path:
            continue
        if 'results' not in data or not data['results']:
            print(f"No results in {file_path}; skipping")
            continue
        
        wrong_counts = []
        for result in data['results']:
            if 'wrong_count' in result:
                wrong_counts.append(result['wrong_count'])
                y_label="Wrong Count"
            elif 'accuracy' in result:
                wrong_counts.append(result['accuracy'])
                y_label="Accuracy"
        
        if not wrong_counts:
            print(f"No wrong_count/accuracy in {file_path}; skipping")
            continue
        
        # Convert to numpy for stats.
        wrong_counts_array = np.array(wrong_counts)
        
        # Mean/std over samples.
        means = np.mean(wrong_counts_array, axis=0)
        stds = np.std(wrong_counts_array, axis=0)
        
        # Label
        label = extract_file_label(file_path, data)
        
        stats[label] = {
            'means': means,
            'stds': stds,
            'file_path': file_path,
            'num_samples': len(wrong_counts)
        }
    
    return stats, y_label

# Plot lines with a std band.
def plot_lines(stats, output_filename='wrong_count_comparison.png', y_label='Accuracy'):
    plt.figure(figsize=(12, 8))
    
    # Colors and markers (up to 4 groups).
    colors = ['blue', 'red', 'green', 'orange']
    markers = ['o', 's', '^', 'D']
    
    for i, (label, data) in enumerate(stats.items()):
        x = np.arange(len(data['means']))
        means = data['means']
        stds = data['stds']
        # Line.
        plt.plot(x, means, label=label, color=colors[i % len(colors)], marker=markers[i % len(markers)], 
                linewidth=2, markersize=6)
        
        # Std band.
        plt.fill_between(x, means - stds, means + stds, 
                        alpha=0.2, color=colors[i % len(colors)])
    
    # Styling.
    plt.xlabel('Rounds', fontsize=15)
    plt.ylabel(f'{y_label}', fontsize=15)
    plt.title(f'{y_label} Comparison with Standard Deviation', fontsize=16)
    plt.legend(fontsize=15)
    plt.grid(True, alpha=0.3)
    plt.xticks(x, fontsize=15)
    plt.yticks(fontsize=15)
    
    # Save.
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.show()

def parse_arguments():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot wrong-count/accuracy lines")
    parser.add_argument("--root_dir", type=str, required=True, 
                       help="Root directory (JSON files are discovered recursively)")
    parser.add_argument("--output", type=str, default="wrong_count_comparison.png",
                       help="Output image filename")
    return parser.parse_args()

def main():
    # Parse args.
    args = parse_arguments()
    
    # Validate root dir.
    if not os.path.exists(args.root_dir):
        print(f"Error: root dir '{args.root_dir}' does not exist")
        return
    
    # Read data.
    print(f"Reading JSON files under '{args.root_dir}' ...")
    data_dict = read_json_files_from_directory(args.root_dir)
    
    if not data_dict:
        print("Error: no valid JSON files found")
        return
    
    print(f"Loaded {len(data_dict)} JSON files")
    
    # Stats.
    stats, y_label = calculate_stats(data_dict)
    
    if not stats:
        print("Error: no files with wrong_count/accuracy found")
        return
    
    # Print stats.
    print("\nStats:")
    for label, data in stats.items():
        print(f"\n{label}:")
        print(f"  File: {data['file_path']}")
        print(f"  Samples: {data['num_samples']}")
        print(f"  Mean: {data['means']}")
        print(f"  Std: {data['stds']}")
    
    # Plot.
    plot_lines(stats, args.output, y_label)
    print(f"\nSaved plot to '{args.output}'")

if __name__ == "__main__":
    main()
