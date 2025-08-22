#!/bin/bash

CUSTOM_MODEL_PATH=$1
iter_num=3
for i in $(seq 1 $iter_num); do
    echo "Running Knowledge Distillation Iter ${i}"
    if [ "$i" -eq 1 ]; then
        MODEL=$CUSTOM_MODEL_PATH
    else
        MODEL=$OUTPUT_DIR
    fi
    OUTPUT_DIR="checkpoints/gpt2-kd-qwen-dolly-iter${i}"
    PROMPT="d:/Python/GenAI/DSKD/data/dolly/train.jsonl"
    OUT="kd-gpt2-qwen-dolly-iter${i}"
    DATASET="synthetic_data_kd_gpt2_qwen_dolly_iter${i}_score"

    echo "Generating responses with GPT-2 student model..."
    bash scripts/generate.sh --model $MODEL --prompt $PROMPT --out_path $OUT --knowledge_distillation
    
    echo "Ranking responses with Qwen 1.5 1.8B teacher model..."
    python3 scripts/rank.py \
        --model gpt2 \
        --output_dir $OUT \
        --prompts $PROMPT \
        --pairs 5 \
        --frac_len 2600 \
        --data_frac 0 \
        --gpu 0 \
        --use_teacher_llm \
        --teacher_model Qwen/Qwen1.5-1.8B \
        --batch_size 8 \
        --bt_conversion_method bradley_terry_mle
    
    echo "Computing probabilities and preparing dataset..."
    python3 scripts/compute_prob.py \
        --gpu_ids "0,1,2,3,4,5,6,7" \
        --output_dir $OUT \
        --pairs 5 \
        --frac_len 2600 \
        --prompts $PROMPT
    
    echo "Training GPT-2 student model with SPPO..."
    bash scripts/pipeline.sh \
        --model $MODEL \
        --iter $i \
        --dataset $DATASET \
        --output_dir $OUTPUT_DIR \
        --num 1 \
        --learning_rate 1.0e-6 \
        --batch_size 16
done

echo "Knowledge distillation complete!"
