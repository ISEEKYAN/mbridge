
MCORE="../3rdparty/Megatron-LM/"
export PYTHONPATH="$PWD:$MCORE:$PYTHONPATH"

# model_path="/home/hf-hub/Qwen/Qwen3-Next-80B-A3B-Instruct/"
model_path="/home/hf_reconverted_release"
torchrun --nproc_per_node=8 example/qwen3_next/load_model_and_forward.py \
    --model_path $model_path \
    --tp 2 \
    --pp 2 \
    --ep 4 \
    --etp 1