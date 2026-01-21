#!/bin/bash

# Train Qwen3-8B EAGLE3 10-layer draft model from scratch
# 10 layers = ~2.36B trainable parameters
# Extended TTT length and higher loss decay factor for better long-range prediction

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
    --draft-model-config $ROOT_DIR/configs/qwen3-8b-eagle3-10layer.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen3-8b-eagle3-10layer \
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
    --wandb-key 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 \
    --wandb-project qwen3-8b-eagle3-10layer \
    --wandb-name 10layer-ttt12-decay09


