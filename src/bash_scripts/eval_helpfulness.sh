#!/bin/bash -l

# Input file should be the output of the CCP evaluation.
INPUT_FILE=""
# Reference file that INPUT_FILE is compared to.
REF_FILE=""
OUTPUT_FILE="" # have to end with .xlsx
DATASET="bio" # must be bio or wildhalu
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_file)
            INPUT_FILE="$2"
            shift 2
            ;;
        --ref_file)
            REF_FILE="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

CACHE_NAME="helpfulness_cache"

# The printed Score is the helpfulness score. Evaluation details will be saved in OUTPUT_FILE.
python ../compute_helpfulness.py --assistant_a $REF_FILE --assistant_b $INPUT_FILE --data $DATASET --cache_name $CACHE_NAME --output_file $OUTPUT_FILE --real_time
