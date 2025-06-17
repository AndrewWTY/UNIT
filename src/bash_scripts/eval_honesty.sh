#!/bin/bash -l

# Input file should be the output of the CCP evaluation.
INPUT_FILE=""
OUTPUT_FILE=""
DATABASE="enwiki" # must be enwiki or wildhalu
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_file)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --database)
            DATABASE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done



# If the fact-check database is not downloaded, download it.
if [ ! -f "fact_check_cache/enwiki-20230401.db" ]; then
    cd ../factcheck_cache
    python download_fact_score_db.py
    cd ../src
fi

INPUT_FILE_PROMPT=$(echo "${INPUT_FILE}" | sed 's/.jsonl/.json/')
FC_CACHE="fact_check_cache"

python ../factcheck.py --input_file $INPUT_FILE --prompt_file $INPUT_FILE_PROMPT  --llm gpt-4o-mini --cache_name $FC_CACHE --fc_real_time --sum_real_time --output_file $OUTPUT_FILE --database $DATABASE


# Calculate the truthfulness and honesty scores. The printed Overall Acc. is the truthfulness score. The printed Macro Avg. is the honesty balanced accuracy score.
python ../compute_truthfulness.py --input_file $OUTPUT_FILE --setting $SETTING_NAME

# Compute the upper bound of the honesty balanced accuracy score. 
#The printed value is the upper bound of the honesty balanced accuracy score, given the ground truth CCP and the CCP threshold searched across all possible values.
python ../compute_honesty_upper_bound.py --input_file $OUTPUT_FILE --setting $SETTING_NAME