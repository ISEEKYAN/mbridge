#!/bin/bash

MODEL_PATH=${MODEL_PATH:-"zai-org/glm-4-9b"}

python example/0.load_model_and_generate_single_gpu.py \
    --model_path $MODEL_PATH

python example/1.load_model_and_export_single_gpu.py \
    --model_path $MODEL_PATH