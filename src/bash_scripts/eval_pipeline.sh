#!/bin/bash

set -e
# ------------------------------
# Default values
# ------------------------------
QUANTILE=''
FINETUNE_METHOD='sft'  # Now expected to be sft, orpo, or dpo
BASE_MODEL="Llama-3.1-8B"  # New flag for base model naming
MODEL_PRESET=''          # New flag to override the old preset logic
TRAIN_DATA='no'         # default train_data (can be "lima", "ragqa", or "limaragqa")
MODEL_NAME=""
TEST_DATA="bio"         # can be "bio", "wildhalu"
NORMAL_PROMPT=false
GRID_SEARCH=false
BASELINE5=false
RAGQARATIO=''
GET_CCP_FROM_RESPONSE=false
USER_SPECIFIED_FILE=""
SURGERY_MODEL='llama'
ONLY_INFERENCE=false
CONFIG_FILE="../inference_configs/eval_inference_config.yaml"

# ------------------------------
# Ensure required folder layout exists
# ------------------------------
for dir in \
    "../inference_configs" \
    "../data_to_probe_ccp" \
    "../evaluate_database" \
    "../pipeline_status_database"
do
    [ -d "$dir" ] || mkdir -p "$dir"
done

# ------------------------------
# Initialize step flags
# ------------------------------
RUN_INFERENCE=false
RUN_CCP_EXTRACT=false
RUN_CCP_MATCH=false
RUN_CCP_FINAL=false
RUN_COMPUTE_CCP=false
RUN_EVALUATE_CALIBRATION=false

# Track if any step arguments were passed
STEP_ARGS_PROVIDED=false

# ------------------------------
# Parse command-line arguments
# ------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --quantile)
            QUANTILE="$2"
            shift 2
            ;;
        --finetune_method)
            FINETUNE_METHOD="$2"
            shift 2
            ;;
        --base_model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --train_data)
            TRAIN_DATA="$2"
            shift 2
            ;;
        --test_data)
            TEST_DATA="$2"
            shift 2
            ;;
        --normal_prompt)
            NORMAL_PROMPT=true
            shift
            ;;
        --inference)
            RUN_INFERENCE=true
            STEP_ARGS_PROVIDED=true
            shift
            ;;
        --extract)
            RUN_CCP_EXTRACT=true
            STEP_ARGS_PROVIDED=true
            shift
            ;;
        --match)
            RUN_CCP_MATCH=true
            STEP_ARGS_PROVIDED=true
            shift
            ;;
        --final)
            RUN_CCP_FINAL=true
            STEP_ARGS_PROVIDED=true
            shift
            ;;
        --compute_ccp)
            RUN_COMPUTE_CCP=true
            STEP_ARGS_PROVIDED=true
            shift
            ;;
        --evaluate_calibration)
            RUN_EVALUATE_CALIBRATION=true
            STEP_ARGS_PROVIDED=true
            shift
            ;;
        --no_eval)
            RUN_EVALUATE_CALIBRATION=false
            STEP_ARGS_PROVIDED=true
            shift
            ;;
        --grid_search)
            GRID_SEARCH=true
            shift
            ;;
        --ragqa_ratio)
            RAGQARATIO="$2"
            shift 2
            ;;
        --surgery_model)
            SURGERY_MODEL="$2"
            shift 2
            ;;
        --get_ccp_from_response)
            GET_CCP_FROM_RESPONSE=true
            USER_SPECIFIED_FILE="$2"
            shift 2
            ;;
        --only_inference)
            ONLY_INFERENCE=true
            RUN_INFERENCE=true
            STEP_ARGS_PROVIDED=true
            shift
            ;;
        *)
            echo "Unknown parameter passed: $1"
            echo "Usage: bash script.sh [--quantile QUANTILE] [--finetune_method FINETUNE_METHOD] [--base_model BASE_MODEL] [--model_name MODEL_NAME] [--train_data TRAIN_DATA] [--test_data TEST_DATA] [--normal_prompt] [--inference] [--extract] [--match] [--final] [--compute_ccp] [--evaluate_calibration] [--get_ccp_from_response FILEPATH] [--only_inference]"
            exit 1
            ;;
    esac
done

# ------------------------------
# If --only_inference passed, ensure only inference is run
# ------------------------------
if [ "$ONLY_INFERENCE" = true ]; then
    RUN_CCP_EXTRACT=false
    RUN_CCP_MATCH=false
    RUN_CCP_FINAL=false
    RUN_COMPUTE_CCP=false
    RUN_EVALUATE_CALIBRATION=false
    GET_CCP_FROM_RESPONSE=false
fi

# ------------------------------
# If no step arguments were provided, run all steps
# ------------------------------
if [ "$STEP_ARGS_PROVIDED" = false ]; then
    RUN_INFERENCE=true
    RUN_CCP_EXTRACT=true
    RUN_CCP_MATCH=true
    RUN_CCP_FINAL=true
    RUN_COMPUTE_CCP=true
    RUN_EVALUATE_CALIBRATION=true
fi

# ------------------------------
# If user provided --get_ccp_from_response, skip inference and override INPUT_FILE
# ------------------------------
if [ "$GET_CCP_FROM_RESPONSE" = true ]; then
    RUN_INFERENCE=false
    # We'll still run the subsequent steps. If user didn't specify them, they will be automatically set to "all steps" anyway.
fi

# ------------------------------
# Determine model name (if not user-supplied)
# ------------------------------
if [ -z "$MODEL_NAME" ]; then
    MODEL_NAME="/home/storage/data/${BASE_MODEL}-${FINETUNE_METHOD}-${SURGERY_MODEL}-q${QUANTILE}-${TRAIN_DATA}"
fi

if [ "$RAGQARATIO" != "" ]; then
    MODEL_NAME="${MODEL_NAME}-${RAGQARATIO}"
fi

# ------------------------------
# Determine dataset path based on TEST_DATA
# ------------------------------
if [ "$TEST_DATA" = "bio" ]; then
    DATASET_PATH="../inference_data/biography_prompt.jsonl"
elif [ "$TEST_DATA" = "wildhalu" ]; then
    DATASET_PATH="../inference_data/wildhalu_prompt.jsonl"
elif [ "$GET_CCP_FROM_RESPONSE" = true ]; then
    # If user provided --get_ccp_from_response, we don't need to specify a dataset path
    echo "Skipping dataset path specification as per user request."
else
    echo "Invalid value for --test_data: $TEST_DATA"
    echo "Must be either 'bio' or 'wildhalu' or 'gsm8k'"
    exit 1
fi

# ------------------------------
# Construct the cache name
# ------------------------------
if [ "$GET_CCP_FROM_RESPONSE" = true ]; then
    # Remove any trailing .jsonl so we don’t get abcd.jsonl_calibrated.jsonl
    CACHE_NAME="$(basename "$USER_SPECIFIED_FILE" .jsonl)"
else
    CACHE_NAME="${TRAIN_DATA}_q${QUANTILE}_${FINETUNE_METHOD}_${TEST_DATA}"
fi

if [ "$NORMAL_PROMPT" = true ]; then
    CACHE_NAME="${CACHE_NAME}_normal_prompt"
fi
if [ "$RAGQARATIO" != "" ]; then
    CACHE_NAME="${CACHE_NAME}_${RAGQARATIO}"
fi

if [ -z "$MODEL_NAME" ]; then
    CACHE_NAME="${CACHE_NAME}_${SURGERY_MODEL}_${BASE_MODEL}"
else
    # Keep only the part after the final “/”  →  e.g. home/checkpoints/llama-checkpoint1 → llama-checkpoint1
    MODEL_BASE="${MODEL_NAME##*/}"
    CACHE_NAME="${CACHE_NAME}_${MODEL_BASE}"
fi

# CACHE_NAME="${CACHE_NAME}_${SURGERY_MODEL}_${BASE_MODEL}"s

echo "CACHE_NAME: $CACHE_NAME"

# ------------------------------
# Choose system prompt
# ------------------------------
if [ "$NORMAL_PROMPT" = true ]; then
    SYS_MESSAGE="You are a helpful assistant.you should answer user's query directly, providing a helpful and accurate response to the query."
else
    SYS_MESSAGE="You are a helpful assistant.you should answer user's query first, providing a helpful and accurate response.Then write a <reflection> section following your response, listing all the factual claims you made in your response that you are uncertain about.\n\nOutput your reflection in the following format ONLY:\n<reflection>\nThe following summarizes the facts that I am uncertain about in my answer:\n1. [factual claim 1 that you are uncertain about]\n2. [factual claim 2 that you are uncertain about]\n3. [factual claim 3 that you are uncertain about]\n...[more factual claims]..."
fi

# ------------------------------
# Print out the chosen parameters
# ------------------------------
echo "Using QUANTILE: $QUANTILE"
echo "Using FINETUNE_METHOD: $FINETUNE_METHOD"
echo "Using TRAIN_DATA: $TRAIN_DATA"
echo "Using TEST_DATA: $TEST_DATA"
echo "DATASET_PATH: $DATASET_PATH"
echo "RAGQARATIO: $RAGQARATIO"
echo "Using MODEL_NAME: $MODEL_NAME"
echo "Using CONFIG_FILE: $CONFIG_FILE"
echo "CACHE_NAME: $CACHE_NAME"
echo "GET_CCP_FROM_RESPONSE: $GET_CCP_FROM_RESPONSE"


echo "Steps to execute:"
[ "$RUN_INFERENCE" = true ] && echo " - Inference"
[ "$RUN_CCP_EXTRACT" = true ] && echo " - CCP Extract"
[ "$RUN_CCP_MATCH" = true ] && echo " - CCP Match"
[ "$RUN_CCP_FINAL" = true ] && echo " - CCP Final Output"
[ "$RUN_COMPUTE_CCP" = true ] && echo " - Compute CCP"
[ "$RUN_EVALUATE_CALIBRATION" = true ] && echo " - Evaluate Calibration"

# ----------------------------------
# Function to check GPU availability
# ----------------------------------
check_gpus() {
    read -r gpu_id _ <<< $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' '$2 < 35000' | sort -t',' -k2 -n | head -n1)
    gpu_array=($gpu_id)
}

# ------------------------------
# Define paths and filenames
# ------------------------------
SCRIPT_PATH="../inference_pipeline.py"
INFERENCE_DIR="../data_to_probe_ccp/"
EVALUATE_DIR="../evaluate_database/"

# If user did NOT provide --get_ccp_from_response, set the default output for inference
if [ "$GET_CCP_FROM_RESPONSE" = false ]; then
    INPUT_FILE="${INFERENCE_DIR}${CACHE_NAME}.jsonl"
else
    # If user provided the flag, override INPUT_FILE with the user-specified file
    INPUT_FILE="${INFERENCE_DIR}$USER_SPECIFIED_FILE"
fi
echo "INPUT_FILE: $INPUT_FILE"

PROMPT_NAME="prompt"
RESPONSE_NAME="responses"

EXTRACT_FILE_NAME="${EVALUATE_DIR}${CACHE_NAME}_extract.json"
MATCH_FILE_NAME="${EVALUATE_DIR}${CACHE_NAME}_match.json"
FINAL_FILE_NAME="${EVALUATE_DIR}${CACHE_NAME}_final.json"

CCP_OUTPUT_FILE_NAME="${EVALUATE_DIR}ccp_only/${CACHE_NAME}_ccp.json"
CCP_CACHE_FILE="${EVALUATE_DIR}ccp_only/${CACHE_NAME}_ccp_cache.json"
CALIBRATION_OUTPUT_FILE_NAME="${EVALUATE_DIR}calibration_result/${CACHE_NAME}_calibrated.jsonl"

# ------------------------------
# Step completion checks
# ------------------------------
step_completed() {
    local step="$1"
    grep -Fxq "$step" "$STATUS_FILE"
}

STATUS_DIR="../pipeline_status_database/"
mkdir -p "$STATUS_DIR"
echo "now using $CACHE_NAME"
STATUS_FILE="${STATUS_DIR}/pipeline_status_${CACHE_NAME}.txt"
if [ ! -f "$STATUS_FILE" ]; then
    touch "$STATUS_FILE"
fi

# ------------------------------
# 1) INFERENCE
# ------------------------------
if [ "$RUN_INFERENCE" = true ]; then
    if step_completed "inference"; then
        echo "Step 'inference' already completed for $CACHE_NAME. Skipping..."
    else
        echo "Starting inference step..."
        source ../vllm/bin/activate
        # ------------------------------
        # Wait for GPU availability
        # ------------------------------
        while true; do
            check_gpus
            if [ ${#gpu_array[@]} -ge 1 ]; then
                export CUDA_VISIBLE_DEVICES="${gpu_array[0]}"
                echo "Using GPU: $CUDA_VISIBLE_DEVICES"
                break
            else
                echo "Waiting for GPUs with memory usage below 26000 MiB..."
                echo "Current GPUs id below threshold: [$available_gpus]"
                sleep 30
            fi
        done

        python3 "$SCRIPT_PATH" \
            --config "$CONFIG_FILE" \
            --output_file "$INPUT_FILE" \
            --model_type "$MODEL_NAME" \
            --system_prompt "$SYS_MESSAGE" \
            --dataset_path "$DATASET_PATH"

        deactivate
        echo "inference" >> "$STATUS_FILE"
    fi
else
    echo "Skipping inference step as per user request or because --get_ccp_from_response was used."
fi

# ------------------------------
# 2) CCP EXTRACT
# ------------------------------

if [ "$RUN_CCP_EXTRACT" = true ]; then
    source ../ccp/bin/activate

    if step_completed "ccp_extract"; then
        echo "Step 'ccp_extract' already completed for $CACHE_NAME. Skipping..."
    else
        echo "Starting ccp_extract step..."
        python ../extract_sentence_facts.py \
            --input_file "$INPUT_FILE" \
            --output_file ragqa_sent_claims.json \
            --cache_name "$CACHE_NAME" \
            --prompt_column "$PROMPT_NAME" \
            --response_column "$RESPONSE_NAME" \
            --has_reflection
        echo "ccp_extract" >> "$STATUS_FILE"
    fi

    if step_completed "ccp_extract_check"; then
        echo "Step 'ccp_extract_check' already completed for $CACHE_NAME. Skipping..."
    else
        echo "Starting ccp_extract_check step..."
        python ../check_openai_batch.py \
            --output_file "$EXTRACT_FILE_NAME" \
            --cache_name "$CACHE_NAME" \
            --input_file "$INPUT_FILE" \
            --extract
        echo "ccp_extract_check" >> "$STATUS_FILE"
    fi

    deactivate
else
    echo "Skipping CCP extract steps as per user request."
fi

# ------------------------------
# 3) CCP MATCH
# ------------------------------
if [ "$RUN_CCP_MATCH" = true ]; then
    source ../ccp/bin/activate

    if step_completed "ccp_match"; then
        echo "Step 'ccp_match' already completed for $CACHE_NAME. Skipping..."
    else
        echo "Starting ccp_match step..."
        python ../extract_sentence_facts.py \
            --input_file "$INPUT_FILE" \
            --output_file ../ragqa_sent_claims.json \
            --cache_name "$CACHE_NAME" \
            --prompt_column "$PROMPT_NAME" \
            --response_column "$RESPONSE_NAME" \
            --has_reflection \
            --extract_batch_result "$EXTRACT_FILE_NAME"
        echo "ccp_match" >> "$STATUS_FILE"
    fi

    if step_completed "ccp_match_check"; then
        echo "Step 'ccp_match_check' already completed for $CACHE_NAME. Skipping..."
    else
        echo "Starting ccp_match_check step..."
        python ../check_openai_batch.py \
            --output_file "$MATCH_FILE_NAME" \
            --cache_name "$CACHE_NAME" \
            --input_file "$INPUT_FILE" \
            --match
        echo "ccp_match_check" >> "$STATUS_FILE"
    fi

    deactivate
else
    echo "Skipping CCP match steps as per user request."
fi

# ------------------------------
# 4) CCP FINAL OUTPUT
# ------------------------------
if [ "$RUN_CCP_FINAL" = true ]; then
    source ../ccp/bin/activate

    if step_completed "ccp_final_output"; then
        echo "Step 'ccp_final_output' already completed for $CACHE_NAME. Skipping..."
    else
        echo "Starting ccp_final_output step..."
        python ../extract_sentence_facts.py \
            --input_file "$INPUT_FILE" \
            --output_file "$FINAL_FILE_NAME" \
            --cache_name "$CACHE_NAME" \
            --prompt_column "$PROMPT_NAME" \
            --response_column "$RESPONSE_NAME" \
            --extract_batch_result "$EXTRACT_FILE_NAME" \
            --match_batch_result "$MATCH_FILE_NAME" \
            --has_reflection
        echo "ccp_final_output" >> "$STATUS_FILE"
    fi

    deactivate
else
    echo "Skipping CCP final output step as per user request."
fi

# ------------------------------
# 5) COMPUTE CCP
# ------------------------------
if [ "$RUN_COMPUTE_CCP" = true ]; then
    if step_completed "compute_ccp"; then
        echo "Step 'compute_ccp' already completed for $CACHE_NAME. Skipping..."
    else
        echo "Starting compute_ccp step..."
        source ../ccp/bin/activate

        # Check GPU availability again
        while true; do
            check_gpus
            if [ ${#gpu_array[@]} -ge 1 ]; then
                export CUDA_VISIBLE_DEVICES="${gpu_array[0]}"
                echo "Using GPU: $CUDA_VISIBLE_DEVICES"
                break
            else
                echo "Waiting for GPUs with memory usage below 40000 MiB..."
                echo "Current GPUs id below threshold: [$available_gpus]"
                sleep 30
            fi
        done
        echo "Model name: $MODEL_NAME"
        echo "CCP output file base: $CCP_OUTPUT_FILE_NAME"
        python ../compute_ccp.py \
            --input_file "$INPUT_FILE" \
            --output_file "$CCP_OUTPUT_FILE_NAME" \
            --prompt_column "$PROMPT_NAME" \
            --response_column "$RESPONSE_NAME" \
            --cache_file "$CCP_CACHE_FILE" \
            --nli_context fact_pref \
            --claim_file "$FINAL_FILE_NAME" \
            --model_name "$MODEL_NAME"


        echo "compute_ccp" >> "$STATUS_FILE"
        deactivate
    fi
else
    echo "Skipping compute CCP step as per user request."
fi



# ------------------------------
# 6) EVALUATE CALIBRATION
# ------------------------------
if [ "$RUN_EVALUATE_CALIBRATION" = true ]; then
    source ../ccp/bin/activate

    echo "Starting evaluate_calibration step (overriding status check)..."

    # Depending on the surgery model, select appropriate threshold declarations.
    if [ "$SURGERY_MODEL" = "qwen14b" ] || [ "$SURGERY_MODEL" = "qwen" ]; then
        # Qwen-specific thresholds
        declare -A CCP_THRESHOLDS_LIMA_QWEN
        CCP_THRESHOLDS_LIMA_QWEN["50"]=-0.249725
        CCP_THRESHOLDS_LIMA_QWEN["65"]=-0.106227
        CCP_THRESHOLDS_LIMA_QWEN["75"]=-0.044809
        CCP_THRESHOLDS_LIMA_QWEN["85"]=-0.010692
        CCP_THRESHOLDS_LIMA_QWEN["95"]=-0.000484
        CCP_THRESHOLDS_LIMA_QWEN["no-surgery"]=-0.044809

        declare -A CCP_THRESHOLDS_RAGQA_QWEN
        CCP_THRESHOLDS_RAGQA_QWEN["50"]=-0.05710210
        CCP_THRESHOLDS_RAGQA_QWEN["65"]=-0.01201692
        CCP_THRESHOLDS_RAGQA_QWEN["75"]=-0.002515925
        CCP_THRESHOLDS_RAGQA_QWEN["85"]=-0.0002063033
        CCP_THRESHOLDS_RAGQA_QWEN["95"]=-0.0000004672050
        CCP_THRESHOLDS_RAGQA_QWEN["no-surgery"]=-0.002515925
        

        declare -A CCP_THRESHOLDS_LIMARAGQA_QWEN
        CCP_THRESHOLDS_LIMARAGQA_QWEN["50"]=-0.06446606
        CCP_THRESHOLDS_LIMARAGQA_QWEN["65"]=-0.01421335
        CCP_THRESHOLDS_LIMARAGQA_QWEN["75"]=-0.003058444
        CCP_THRESHOLDS_LIMARAGQA_QWEN["85"]=-0.0002604213
        CCP_THRESHOLDS_LIMARAGQA_QWEN["95"]=-0.00000062264
        CCP_THRESHOLDS_LIMARAGQA_QWEN["no-surgery"]=-0.003058444


        case "$TRAIN_DATA" in
            "lima")
                CCP_THRESHOLDS=CCP_THRESHOLDS_LIMA_QWEN
                ;;
            "ragqa")
                CCP_THRESHOLDS=CCP_THRESHOLDS_RAGQA_QWEN
                ;;
            "limaragqa"|"no")
                CCP_THRESHOLDS=CCP_THRESHOLDS_LIMARAGQA_QWEN
                ;;
            *)
                echo "Invalid TRAIN_DATA value: $TRAIN_DATA"
                exit 1
                ;;
        esac
    # else
    elif [ "$SURGERY_MODEL" = "llama" ]; then
        # Llama thresholds (existing values)
        declare -A CCP_THRESHOLDS_LIMA
        CCP_THRESHOLDS_LIMA["50"]=-0.217175
        CCP_THRESHOLDS_LIMA["65"]=-0.086788
        CCP_THRESHOLDS_LIMA["75"]=-0.037325
        CCP_THRESHOLDS_LIMA["85"]=-0.008926
        CCP_THRESHOLDS_LIMA["95"]=-0.000382
        CCP_THRESHOLDS_LIMA["no-surgery"]=-0.037325
        CCP_THRESHOLDS_LIMA["instruct"]=-0.037325

        declare -A CCP_THRESHOLDS_RAGQA
        CCP_THRESHOLDS_RAGQA["50"]=-0.052052
        CCP_THRESHOLDS_RAGQA["65"]=-0.011424
        CCP_THRESHOLDS_RAGQA["75"]=-0.002476
        CCP_THRESHOLDS_RAGQA["85"]=-0.000260
        CCP_THRESHOLDS_RAGQA["95"]=-0.000005
        CCP_THRESHOLDS_RAGQA["no-surgery"]=-0.002476
        CCP_THRESHOLDS_RAGQA["instruct"]=-0.002476

        declare -A CCP_THRESHOLDS_LIMARAGQA
        CCP_THRESHOLDS_LIMARAGQA["50"]=-0.093923
        CCP_THRESHOLDS_LIMARAGQA["65"]=-0.025266
        CCP_THRESHOLDS_LIMARAGQA["75"]=-0.006779
        CCP_THRESHOLDS_LIMARAGQA["85"]=-0.000817
        CCP_THRESHOLDS_LIMARAGQA["95"]=-0.000015
        CCP_THRESHOLDS_LIMARAGQA["no-surgery"]=-0.006779
        CCP_THRESHOLDS_LIMARAGQA["instruct"]=-0.037325

        case "$TRAIN_DATA" in
            "lima")
                CCP_THRESHOLDS=CCP_THRESHOLDS_LIMA
                ;;
            "ragqa")
                CCP_THRESHOLDS=CCP_THRESHOLDS_RAGQA
                ;;
            "limaragqa"|"no")
                CCP_THRESHOLDS=CCP_THRESHOLDS_LIMARAGQA
                ;;
            *)
                echo "Invalid TRAIN_DATA value: $TRAIN_DATA"
                exit 1
                ;;
        esac
    fi

    eval "ccp_threshold=\${${CCP_THRESHOLDS}[$QUANTILE]}"

    if [ -z "$ccp_threshold" ]; then
        echo "CCP threshold for quantile=$QUANTILE and train_data=$TRAIN_DATA not found."
        exit 1
    else
        echo "Using ccp_threshold: $ccp_threshold"
    fi

    if [ "$GRID_SEARCH" = true ]; then
        python ../evaluate_calibration.py \
            --input_file "$CCP_OUTPUT_FILE_NAME" \
            --output_file "$CALIBRATION_OUTPUT_FILE_NAME" \
            --ccp_threshold "$ccp_threshold" \
            --grid_search

    else
        python ../evaluate_calibration.py \
            --input_file "$CCP_OUTPUT_FILE_NAME" \
            --output_file "$CALIBRATION_OUTPUT_FILE_NAME" \
            --ccp_threshold "$ccp_threshold"
    fi

    echo "evaluate_calibration" >> "$STATUS_FILE"
    deactivate
else
    echo "Skipping evaluate calibration step as per user request."
fi