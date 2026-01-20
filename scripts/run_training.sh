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


echo "Step 4: Training GNN model with multi-branch architecture..." # only for ours/gsafeguard
python ./train/trainer.py --epochs 50 --batch_size 32 --lr 0.001 --save_dir $CHECKPOINT_DIR --attack_mode $ATTACK_MODE --dataset $DATASET --name $NAME --selective_training --guard $GUARD_TYPE

