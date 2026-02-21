#!/bin/bash
ps -ef | grep python | awk  '{print $2}' | xargs -I {} kill -9 {}
sleep 1

MLM_PATH="../3rdparty/Megatron-LM/"

export PYTHONPATH=$PWD:$MLM_PATH:$PYTHONPATH
echo "PYTHONPATH ${PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_OFFLINE=1
export GLOO_SOCKET_IFNAME=bond1
export NCCL_SOCKET_IFNAME=bond1

readonly GPUS_PER_NODE=8
readonly NODE_RANK="${OMPI_COMM_WORLD_RANK:-0}"
readonly NNODES="${OMPI_COMM_WORLD_SIZE:-1}"
readonly WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
readonly MASTER_PORT=65535
export MASTER_ADDR="${_MASTER_ADDR:-localhost}"

readonly TP_SIZE=2
readonly PP_SIZE=1
readonly CP_SIZE=1
readonly EP_SIZE=16

echo "INFO
__POD_IP__ $__POD_IP__
NODE_RANK $NODE_RANK
NNODES $NNODES
TP_SIZE $TP_SIZE
PP_SIZE $PP_SIZE
CP_SIZE $CP_SIZE
EP_SIZE $EP_SIZE
"

# torch 启动参数
DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

SAMPLE_TYPE="image"

torchrun $DISTRIBUTED_ARGS \
    example/qwen3_5/load_model_and_inference.py \
    --tp $TP_SIZE \
    --pp $PP_SIZE \
    --ep $EP_SIZE \
    --etp 1 \
    --cp $CP_SIZE \
    --model_path hf-hub/Qwen/Qwen3.5-397B-A17B/ \
    --sample_type $SAMPLE_TYPE \

