#!/bin/bash

# Continue training Qwen3-8B EAGLE3 5-layer draft model from checkpoint
# Starting from: outputs/qwen3-8b-eagle3-5layer/epoch_7_step_30000
# With extended TTT length and higher loss decay factor

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

NUM_GPUS=${1:-8}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-8}

# Checkpoint to resume from
CKPT_DIR=$ROOT_DIR/outputs/qwen3-8b-eagle3-5layer/epoch_7_step_30000

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path Qwen/Qwen3-8B \
    --draft-model-config $ROOT_DIR/configs/qwen3-8b-eagle3-5layer.json \
    --ckpt-dir $CKPT_DIR \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen3-8b-eagle3-5layer-extended \
    --num-epochs 10 \
    --batch-size 4 \
    --learning-rate 5e-5 \
    --max-length 2048 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --ttt-length 12 \
    --loss-decay-factor 0.9 \
    --log-interval 100 \
    --save-interval 1000 \
    --target-model-backend sglang \
    --report-to wandb \
    --wandb-key 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 \
    --wandb-project qwen3-8b-eagle3-5layer \
    --wandb-name 5layer-extended-ttt12-decay09

