#!/bin/bash

set -xeuo pipefail

# LongCat Flash 560B - MBridge Weight Load & Export Test
# Configuration: 32 GPUs (4 nodes x 8 GPUs), TP=8, PP=4, EP=8
# Test: Load HF weights into Megatron, export back, verify correctness

export TRANSFORMERS_OFFLINE=1

# ==================== Environment Setup ====================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_LAUNCH_MODE=PARALLEL
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=8
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_SHM_DISABLE=1
export NCCL_IB_TIMEOUT=60
export NCCL_IB_RETRY_CNT=15
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_NVLS_ENABLE=0

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TORCH_NCCL_BLOCKING_WAIT=1

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true

export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0

# ==================== Path Configuration ====================
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
WORKDIR=${WORKDIR:-"/workdir"}

#export MEGATRON_PATH=${MEGATRON_PATH:-"${WORKDIR}/megatron-lm"}
#export PYTHONPATH=${MEGATRON_PATH}:${WORKDIR}/mbridge:${WORKDIR}:${PYTHONPATH:-}
export PYTHONPATH=${WORKDIR}/mbridge:${WORKDIR}:${PYTHONPATH:-}

# ==================== Model Path ====================
MODEL_PATH=${MODEL_PATH:-"/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/RPG/yanhaonan/models/LongCatFlashChat"}

# ==================== NCCL/IB Configuration ====================
SCRIPTS_DIR="/workdir/scripts"

if [ -f "$SCRIPTS_DIR/ibdev2netdev.sh" ]; then
    bash "$SCRIPTS_DIR/ibdev2netdev.sh"
fi

if [ -f "$SCRIPTS_DIR/setup_ib_devices.sh" ]; then
    bash "$SCRIPTS_DIR/setup_ib_devices.sh"
fi

ENV_FILE="${WORKDIR}/setup_ib_devices.sh"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

unset http_proxy
unset https_proxy

# ==================== Cluster Configuration ====================
echo "=========================================="
echo "Parsing cluster configuration..."
echo "=========================================="

AFO_CONFIG_LOADED=false
if [ -n "${AFO_ENV_CLUSTER_SPEC:-}" ]; then
    echo "Detected AFO_ENV_CLUSTER_SPEC, parsing..."

    CLUSTER_INFO=$(python3 << 'EOF'
import os
import json
import sys

try:
    cluster_spec = json.loads(os.environ["AFO_ENV_CLUSTER_SPEC"])
    role = cluster_spec.get("role", "worker")

    if role != "worker":
        sys.exit(1)

    node_rank = int(cluster_spec.get("index", 0))
    nnodes = len(cluster_spec.get(role, []))

    if nnodes > 0:
        master = cluster_spec[role][0]
        master_addr, master_ports = master.split(":")
        master_ports_list = master_ports.split(",")
        master_port = master_ports_list[0]

        print(f"NODE_RANK={node_rank}")
        print(f"NNODES={nnodes}")
        print(f"MASTER_ADDR={master_addr}")
        print(f"MASTER_PORT={master_port}")
    else:
        sys.exit(1)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
EOF
)

    if [ $? -eq 0 ] && [ -n "${CLUSTER_INFO:-}" ]; then
        eval "${CLUSTER_INFO}"
        AFO_CONFIG_LOADED=true
        echo "Loaded from AFO_ENV_CLUSTER_SPEC:"
        echo "  NODE_RANK=${NODE_RANK}"
        echo "  NNODES=${NNODES}"
        echo "  MASTER_ADDR=${MASTER_ADDR}"
        echo "  MASTER_PORT=${MASTER_PORT}"
    fi
fi

# Default values if not loaded from AFO
NNODES=${NNODES:-4}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}

# Auto-detect MASTER_ADDR if not set
if [ -z "${MASTER_ADDR:-}" ] || [ "${MASTER_ADDR}" = "localhost" ]; then
    if [ "${NODE_RANK}" -eq 0 ]; then
        MASTER_ADDR=$(hostname -I | awk '{print $1}')
    else
        HOST_IP=$(hostname -I | awk '{print $1}')
        IP_PREFIX=$(echo $HOST_IP | cut -d'.' -f1-3)
        HOST_LAST_OCTET=$(echo $HOST_IP | cut -d'.' -f4)
        MASTER_LAST_OCTET=$((HOST_LAST_OCTET - NODE_RANK))
        MASTER_ADDR="${IP_PREFIX}.${MASTER_LAST_OCTET}"
    fi
fi

echo "=========================================="
echo "MBridge Launch Test Configuration:"
echo "  Total Nodes: ${NNODES}"
echo "  GPUs per Node: ${N_GPUS_PER_NODE}"
echo "  Total GPUs: $((NNODES * N_GPUS_PER_NODE))"
echo "  Node Rank: ${NODE_RANK}"
echo "  Master Address: ${MASTER_ADDR}"
echo "  Master Port: ${MASTER_PORT}"
echo "  Model Path: ${MODEL_PATH}"
echo "  Parallelism: TP=8, PP=4, EP=8"
echo "=========================================="

# ==================== Launch Test ====================
torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${N_GPUS_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    mbridge/example/longcat_flash/launch_with_ray.py \
    --model_path ${MODEL_PATH} \
    --tp 8 --pp 4 --ep 8

echo ""
echo "=========================================="
echo "Export test completed on Node ${NODE_RANK}!"
echo "=========================================="
