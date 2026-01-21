#!/bin/bash

# Train Qwen3-8B with 5-layer Qwen3-style EAGLE3 draft model
# Uses Qwen3's native architecture (with QK-Norm) instead of Llama's
#
# Key differences from Llama EAGLE:
# - QK-Norm: RMSNorm applied to Q and K projections before RoPE
# - Configurable attention bias (False for Qwen3)
#
# Qwen3-8B dimensions:
# - hidden_size: 4096
# - intermediate_size: 12288
# - num_attention_heads: 32
# - num_key_value_heads: 8 (GQA)
# - head_dim: 128

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

NUM_GPUS=${1:-8}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-8}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path Qwen/Qwen3-8B \
    --draft-model-config $ROOT_DIR/configs/qwen3-8b-qwen3eagle-5layer.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen3-8b-qwen3eagle-5layer \
    --num-epochs 10 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --ttt-length 7 \
    --loss-decay-factor 0.8 \
    --log-interval 100 \
    --save-interval 5000 \
    --target-model-backend sglang \
    --report-to wandb \
    --wandb-project qwen3-8b-qwen3eagle \
    --wandb-name 5layer-ttt7


