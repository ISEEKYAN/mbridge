#!/bin/bash

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

MODEL_PATH=${MODEL_PATH:-"zai-org/GLM-4.5-Air"}

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

torchrun "${DISTRIBUTED_ARGS[@]}" \
    examples/2.load_model_and_generate_multi_gpu.py \
    --model_path $MODEL_PATH \
    --tp 2 \
    --cp 2 \
    --ep 2 \
    --pp 2 \
    --save_path outputs/multi_gpus_test/$MODEL_PATH