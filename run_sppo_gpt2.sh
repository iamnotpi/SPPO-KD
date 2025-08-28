#!/bin/bash
set -e

# Usage: bash run_sppo_gpt2.sh /abs/path/to/your/gpt2

iter_num=3

if [ -n "$1" ]; then
  INITIAL_MODEL="$1"
else
  echo "Please provide the absolute path to your local GPT-2 folder as the first argument."
  echo "Example: bash run_sppo_gpt2.sh /home/you/models/gpt2"
  exit 1
fi

for i in $(seq 1 $iter_num); do
    echo "Running Iter ${i}"
    if [ "$i" -eq 1 ]; then
        MODEL="$INITIAL_MODEL"
    else
        MODEL=$OUTPUT_DIR
    fi
    OUTPUT_DIR="checkpoints/GPT2-SPPO-Iter${i}"
    PROMPT="${HF_ORG:-UCLA-AGI}/data-mistral-7b-instruct-sppo-iter${i}"
    OUT="data-gpt2-sppo-iter${i}"

    # Generate candidates with the target model (GPT-2) and rank with Qwen teacher
    bash scripts/generate.sh \
      --model "$MODEL" \
      --prompt "$PROMPT" \
      --out_path "$OUT" \
      --use_teacher 1 \
      --teacher_model "Qwen/Qwen1.5-1.8B-Chat"

    # Train SPPO using the synthesized dataset
    bash scripts/pipeline.sh \
      --model "$MODEL" \
      --iter $i \
      --dataset "synthetic_data_gpt2-sppo-iter${i}_score" \
      --output_dir $OUTPUT_DIR \
      --num 1
done


