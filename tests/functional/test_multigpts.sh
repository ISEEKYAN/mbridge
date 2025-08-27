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
    example/2.load_model_and_export_multiple_gpus.py \
    --model_path $MODEL_PATH \
    --tp 8 \
    --cp 1 \
    --ep 8 \
    --etp 1 \
    --pp 1 \
    --save_path outputs/multi_gpus_test/$MODEL_PATH