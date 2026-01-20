#!/bin/bash
export OPENAI_API_KEY=''
export BASE_URL=''

export JUDGER_MODEL='gpt-4o-mini'
export JUDGER_OPENAI_API_KEY=''
export JUDGER_BASE_URL=''


export SAFEGUARD_MODEL='gpt-4o'
export SAFEGUARD_BASE_URL=''
export SAFEGUARD_OPENAI_API_KEY=''

ATTACK_MODE=${1:-"PI"} # TA/MA/PI
MODEL=${2:-"gpt-4o-mini"}
NAME=${3:-"gpt-4o-mini"}  # self-defined label for experiments
DATASET=${4:-"csqa"} # PI: gsm8k/csqa/mmlu MA: memory_attack TA: tool_attack
GUARD_TYPE=${5:-"ours"} # gsafeguard/ours/agentsafe/agentxposed-guide/agentxposed-kick/challenger/inspector

echo "=== Starting Memory Attack Full Pipeline with Multi-Branch GNN ==="
if [ -n "$NAME" ]; then
    echo "Using name suffix: $NAME"
fi

# Build directory path with name suffix
AGENT_GRAPH_DIR="./output/output_${NAME}/agent_graph_dataset"
MODEL_TRAINING_DIR="./output/output_${NAME}/ModelTrainingSet"
CHECKPOINT_DIR="./output/output_${NAME}/checkpoint"
RESULT_DIR="./output/output_${NAME}/result"
SUMMARY_DIR="./output/output_${NAME}/summary_results"

if [ -n "$NAME" ]; then
    AGENT_GRAPH_DIR="${AGENT_GRAPH_DIR}_${NAME}"
    MODEL_TRAINING_DIR="${MODEL_TRAINING_DIR}_${NAME}"
    CHECKPOINT_DIR="${CHECKPOINT_DIR}_${NAME}"
    RESULT_DIR="${RESULT_DIR}_${NAME}"
    SUMMARY_DIR="${SUMMARY_DIR}_${NAME}"
fi


echo "Step 5: Generating testing conversation datasets..."
python ./generate_data/gen_graph.py --model_type $MODEL --attack_mode $ATTACK_MODE --phase test --num_nodes 8 --sparsity 0.2 --num_attackers 3 --samples 12 --save_dir $AGENT_GRAPH_DIR --dataset $DATASET
python ./generate_data/gen_graph.py --model_type $MODEL --attack_mode $ATTACK_MODE --phase test --num_nodes 8 --sparsity 0.4 --num_attackers 3 --samples 12 --save_dir $AGENT_GRAPH_DIR --dataset $DATASET
python ./generate_data/gen_graph.py --model_type $MODEL --attack_mode $ATTACK_MODE --phase test --num_nodes 8 --sparsity 0.6 --num_attackers 3 --samples 12 --save_dir $AGENT_GRAPH_DIR --dataset $DATASET
python ./generate_data/gen_graph.py --model_type $MODEL --attack_mode $ATTACK_MODE --phase test --num_nodes 8 --sparsity 0.8 --num_attackers 3 --samples 12 --save_dir $AGENT_GRAPH_DIR --dataset $DATASET
python ./generate_data/gen_graph.py --model_type $MODEL --attack_mode $ATTACK_MODE --phase test --num_nodes 8 --sparsity 1.0 --num_attackers 3 --samples 12 --save_dir $AGENT_GRAPH_DIR --dataset $DATASET

echo "Step 6: Merging testing datasets..."
if [ -n "$NAME" ]; then
    python ./generate_data/dataset_utils/merge_datasets.py --phase test --attack_mode $ATTACK_MODE --root "${AGENT_GRAPH_DIR}/${ATTACK_MODE}/${DATASET}" --dataset $DATASET
else
    python ./generate_data/dataset_utils/merge_datasets.py --phase test --attack_mode $ATTACK_MODE --dataset $DATASET
fi
