#!/bin/bash
export OPENAI_API_KEY='local'
export BASE_URL='http://127.0.0.1:1234/v1'

export JUDGER_MODEL='deepseek/deepseek-r1-0528-qwen3-8b'
export JUDGER_OPENAI_API_KEY='local'
export JUDGER_BASE_URL='http://127.0.0.1:1234/v1'

export SAFEGUARD_MODEL='deepseek/deepseek-r1-0528-qwen3-8b'
export SAFEGUARD_BASE_URL='http://127.0.0.1:1234/v1'
export SAFEGUARD_OPENAI_API_KEY='local'

ATTACK_MODE=${1:-"PI"} # TA/MA/PI
MODEL=${2:-"deepseek/deepseek-r1-0528-qwen3-8b"}
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

echo "Step 7: Running real-time defense evaluation..."
python ./evaluate/main_defense_repair_test.py --graph_type random --gnn_checkpoint_dir $CHECKPOINT_DIR --model_type $MODEL --dataset $DATASET --attack_mode $ATTACK_MODE --num_dialogue_turns 3 --save_dir $RESULT_DIR --name $NAME --guard $GUARD_TYPE 

echo "Step 8: Evaluating output..."
python ./evaluate/evaluate_output.py --attack_mode $ATTACK_MODE --save_dir $RESULT_DIR/$DATASET --summary_dir $SUMMARY_DIR --dataset $DATASET --guard $GUARD_TYPE

echo "Step 9: Generating plots..."
python ./evaluate/plot.py --root_dir $SUMMARY_DIR/$ATTACK_MODE/$DATASET/$GUARD_TYPE --output $SUMMARY_DIR/$ATTACK_MODE/$DATASET/$GUARD_TYPE/comparison.png
