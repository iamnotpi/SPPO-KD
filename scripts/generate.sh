set -e
set -x

export CUDA_VISIBLE_DEVICES="0,1"
AVAILABLE_GPUS=(0 1)
HF_ORG=${HF_ORG:-UCLA-AGI}

MODEL="mistralai/Mistral-7B-Instruct-v0.2"
OUTDIR="data-mistral-7b-instruct-sppo-iter1"

PAIRS=5
FRAC=0
PROMPTS="${HF_PROMPTS_DATASET:-UCLA-AGI/data-mistral-7b-instruct-sppo-iter1}"

# Teacher LLM scoring options (for ranking)
USE_TEACHER=0
TEACHER_MODEL="Qwen/Qwen1.5-1.8B-Chat"
TEACHER_BATCH_SIZE=4
TEACHER_TP=1
BT_METHOD="bradley_terry_mle"
SCORING_TEMPLATE=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
    --pairs)
        PAIRS="$2"
        shift
        ;;
    --frac)
        FRAC="$2"
        shift
        ;;
    --model)
        MODEL="$2"
        shift
        ;;
    --out_path)
        OUTDIR="$2"
        shift
        ;;
    --prompt)
        PROMPTS="$2"
        shift
        ;;
    --use_teacher)
        USE_TEACHER="$2" # 0 or 1
        shift
        ;;
    --teacher_model)
        TEACHER_MODEL="$2"
        shift
        ;;
    --teacher_batch_size)
        TEACHER_BATCH_SIZE="$2"
        shift
        ;;
    --teacher_tp)
        TEACHER_TP="$2"
        shift
        ;;
    --bt_method)
        BT_METHOD="$2"
        shift
        ;;
    --scoring_template)
        SCORING_TEMPLATE="$2"
        shift
        ;;
    *)
        echo "Unknown parameter passed: $1"
        exit 1
        ;;
    esac
    shift
done

#####################
# Generate Data
#####################

#frac length 2600 * num_gpus 8 = 20800, should be larger than the length of the dataset. Change frac_len accordingly when dataset changes

FRAC_LEN=6000
echo "Using frac_len ${FRAC_LEN}"
(
    data_frac=0
    for gpu_id in ${AVAILABLE_GPUS[@]}; do
        CUDA_VISIBLE_DEVICES=$gpu_id python3 scripts/generate.py --model $MODEL --maxlen 2048 --output_dir "generated/$OUTDIR" --prompts $PROMPTS --pairs $PAIRS --world_size 1 --frac_len $FRAC_LEN --data_frac $data_frac > output_log_${gpu_id}.txt 2>&1 &
        ((data_frac+=1));
    done
    wait
) &
all_gen=$!

wait $all_gen

python3 scripts/combine_generate.py --output_dir "generated/$OUTDIR" --gpu_ids "$(IFS=, ; echo "${AVAILABLE_GPUS[*]}")" --pairs $PAIRS


# #####################
# # Rank Data
# #####################

# # frac length 2600 * num_gpus 8 = 20800, should be larger than the length of the dataset. Change frac_len accordingly when dataset changes

python3 scripts/preload.py

(
    data_frac=0
    for gpu_id in ${AVAILABLE_GPUS[@]}; do
        if [ "$USE_TEACHER" -eq 1 ]; then
            CUDA_VISIBLE_DEVICES=$gpu_id python3 scripts/rank.py --model $MODEL --output_dir $OUTDIR --pairs $PAIRS --numgpu ${#AVAILABLE_GPUS[@]} --frac_len $FRAC_LEN --data_frac $data_frac --gpu $gpu_id --prompts $PROMPTS --use_teacher_llm --teacher_model "$TEACHER_MODEL" --batch_size $TEACHER_BATCH_SIZE --tensor_parallel_size $TEACHER_TP --bt_conversion_method $BT_METHOD ${SCORING_TEMPLATE:+--scoring_prompt_template "$SCORING_TEMPLATE"} > rank_log_${gpu_id}.txt 2>&1 &
        else
            CUDA_VISIBLE_DEVICES=$gpu_id python3 scripts/rank.py --model $MODEL --output_dir $OUTDIR --pairs $PAIRS --numgpu ${#AVAILABLE_GPUS[@]} --frac_len $FRAC_LEN --data_frac $data_frac --gpu $gpu_id --prompts $PROMPTS > rank_log_${gpu_id}.txt 2>&1 &
        fi
        ((data_frac+=1));
    done
    wait
) &
all_rank=$!

wait $all_rank

python3 scripts/compute_prob.py --org $HF_ORG --gpu_ids "$(IFS=, ; echo "${AVAILABLE_GPUS[*]}")" --output_dir $OUTDIR --pairs $PAIRS --frac_len $FRAC_LEN --prompts $PROMPTS
