#!/bin/bash -l

MODEL_PATH=""
SETTING_NAME=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --setting_name)
            SETTING_NAME="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done
SYSTEM_PROMPT="You are a helpful assistant.you should answer user’s query directly, providing a helpful and accurate response to the query."
TEMPLATE="{question}"
INST_COLUMN='prompt'
PROMPT_PATH_BIO="../datasets/biography_prompt.jsonl"
PROMPT_PATH_WH="../datasets/wildhalu_prompt_subset.jsonl"
OUTPUT_PATH_BIO="../results/${SETTING_NAME}_bio.csv"
OUTPUT_PATH_WH="../results/${SETTING_NAME}_wildhalu.csv"

# Inference on Biography and WildHalu with the target checkpoint.
# Example output: results/qwen25_limalfrqa01_cut_bio.csv, where limalfraqa01_cut denotes the Qwen2.5 trained on LIMA and 10% of LFRQA with UNIT_CUT.
python ../vllm_inference.py --instruction_field $INST_COLUMN --prompt_file $PROMPT_PATH_BIO --temperature 0 --model_path $MODEL_PATH --output_file $OUTPUT_PATH_BIO --load_tokenizer --system_prompt "$SYSTEM_PROMPT" --template "$TEMPLATE" --gpu_memory_utilization 0.8
python ../vllm_inference.py --instruction_field $INST_COLUMN --prompt_file $PROMPT_PATH_WH --temperature 0 --model_path $MODEL_PATH --output_file $OUTPUT_PATH_WH --load_tokenizer --system_prompt "$SYSTEM_PROMPT" --template "$TEMPLATE" --gpu_memory_utilization 0.8

OUTPUT_PATH_BIO_ATOMIC="../results/${SETTING_NAME}_bio_atomic.jsonl"
OUTPUT_PATH_WH_ATOMIC="../results/${SETTING_NAME}_wildhalu_atomic.jsonl"
# Get atomic claims from the output of the target checkpoint.
# Example output: results/qwen25_limalfrqa01_cut_bio_atomic.jsonl.
python ../get_atomic_claims.py --input_file $OUTPUT_PATH_BIO --data bio --llm gpt-4o --cache_name ${SETTING_NAME}_bio --batch_size 20 --output_file $OUTPUT_PATH_BIO_ATOMIC
python ../get_atomic_claims.py --input_file $OUTPUT_PATH_WH --data wildhalu --llm gpt-4o --cache_name ${SETTING_NAME}_wildhalu --batch_size 20 --output_file $OUTPUT_PATH_WH_ATOMIC


# If the fact-check database is not downloaded, download it.
if [ ! -f "fact_check_cache/enwiki-20230401.db" ]; then
    cd ../factcheck_cache
    python download_fact_score_db.py
    cd ../src
fi

BIO_FC_PROMPT_FILE="../results/${SETTING_NAME}_bio_atomic.json"
BIO_FC_INPUT_FILE=${BIO_FC_PROMPT_FILE}l
BIO_FC_OUTPUT_FILE="results/${SETTING_NAME}_bio_atomic_fc.jsonl"
WH_FC_PROMPT_FILE="../results/${SETTING_NAME}_wildhalu_atomic.json"
WH_FC_INPUT_FILE=${WH_FC_PROMPT_FILE}l
WH_FC_OUTPUT_FILE="../results/${SETTING_NAME}_wildhalu_atomic_fc.jsonl"
FC_CACHE="fact_check_cache"

# Fact-check the extracted atomic claims, example output: results/qwen25_limalfrqa01_cut_bio_atomic_fc.jsonl.
python ../factcheck.py --input_file $BIO_FC_INPUT_FILE --prompt_file $BIO_FC_PROMPT_FILE  --llm gpt-4o-mini --cache_name $FC_CACHE --fc_real_time --sum_real_time --output_file $BIO_FC_OUTPUT_FILE
python ../factcheck.py --input_file $WH_FC_INPUT_FILE --prompt_file $WH_FC_PROMPT_FILE  --llm gpt-4o-mini --cache_name $FC_CACHE --fc_real_time --sum_real_time --output_file $WH_FC_OUTPUT_FILE

# Calculate the truthfulness score. The printed Overall Acc. is the truthfulness score.
python ../compute_truthfulness.py --input_file $BIO_FC_OUTPUT_FILE --setting $SETTING_NAME
python ../compute_truthfulness.py --input_file $WH_FC_OUTPUT_FILE --setting $SETTING_NAME


