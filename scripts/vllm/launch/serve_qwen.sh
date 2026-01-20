#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL=/mnt/shared-storage-user/ai4good1-share/hf_hub/Qwen/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/56e16a623ffb2855ca901a65166a9170e99df127
MAX_MODEL_LEN=65536
MAX_NUM_SEQS=16
NUM_GPUS=4

nohup python scripts/vllm/launch/launch_vllm.py \
    --model $MODEL \
    --tp $NUM_GPUS \
    --max_model_len $MAX_MODEL_LEN \
    --max_num_seqs $MAX_NUM_SEQS > /dev/null 2>&1 &
