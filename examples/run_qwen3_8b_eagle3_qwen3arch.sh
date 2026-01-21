#!/bin/bash

# Train Qwen3-8B EAGLE3 with Qwen3's native architecture (with QK-Norm)
# This uses Qwen3's attention layer structure instead of Llama's

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
    --draft-model-config $ROOT_DIR/configs/qwen3-8b-eagle3-qwen3arch-5layer.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir /shared/yilian/qwen3-8b-eagle/qwen_arch_5layer \
    --num-epochs 10 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --ttt-length 7 \
    --loss-decay-factor 0.8 \
    --log-interval 100 \
    --save-interval 1000 \
    --target-model-backend sglang \
    --report-to wandb \
    --wandb-project qwen3-8b-eagle3-qwen3arch \
    --wandb-name qwen3arch-5layer-ttt7



