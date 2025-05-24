import copy
import json
import os
import random
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Process and split classification JSONL data")
    p.add_argument("--input_file",    type=str,   required=True,   help="Path to the input JSONL file")
    p.add_argument("--output_dir",    type=str,   default="", help="Directory for outputs")
    # p.add_argument("--test_size",     type=float, default=0.0,     help="Fraction of data for test set")
    p.add_argument("--do_plot",       action="store_true",         help="Plot distributions of CCP and max_prob")
    p.add_argument("--all_info",      action="store_true",         help="Include all information-seeking entries")
    p.add_argument("--only_info",     action="store_true",         help="Filter final data to info-seeking only")
    p.add_argument("--no_surgery",    action="store_true",         help="Disable surgery/reflection logic")
    return p.parse_args()

args = parse_args()

INPUT_FILE = args.input_file
OUTPUT_DIR        = args.output_dir
# TEST_SIZE         = args.test_size
DO_PLOT           = args.do_plot
ALL_INFO_SEEKING  = args.all_info
ONLY_OUTPUT_INFO_SEEK = args.only_info
NO_SURGERY        = args.no_surgery

CACHE_NAME = os.path.splitext(os.path.basename(INPUT_FILE))[0]
if args.ragqa_ratio is not None:
    CACHE_NAME += f"_{args.ragqa_ratio}"
if ONLY_OUTPUT_INFO_SEEK:
    CACHE_NAME += "_only_info_seek"
if NO_SURGERY:
    CACHE_NAME += "_no_surgery"


def load_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON object: {e}")
                    print(f"Problematic line: {line}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return data

def save_jsonl(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

def remove_entries_with_none_response(json_data):
    removed_indices = []
    new_data = []
    for i, item in enumerate(json_data):
        if item.get("response") is None:
            removed_indices.append(i)
        else:
            new_data.append(item)
    return new_data, removed_indices

def extract_and_plot_ccp(json_data):
    ccp_values = []
    max_probs = []

    for entry in json_data:
        if 'claim_uncertainty' in entry:
            for uncertainty in entry['claim_uncertainty']:
                if 'ccp' in uncertainty:
                    ccp_values.append(uncertainty['ccp'])
                if 'max_prob' in uncertainty:
                    max_probs.append(uncertainty['max_prob'])

    ccp_series = pd.Series(ccp_values)
    print("Statistical Summary of CCP Values:")
    print(ccp_series.describe())

    max_probs_series = pd.Series(max_probs)
    print("Statistical Summary of Max Probabilities:")
    print(max_probs_series.describe())

    quantiles = [0.75]
    ccp_quantiles = ccp_series.quantile(quantiles)
    max_prob_quantiles = max_probs_series.quantile(quantiles)

    print("\nQuantiles of CCP Values:")
    print(ccp_quantiles)
    print("\nQuantiles of Max_Prob Values:")
    print(max_prob_quantiles)

    print(len(ccp_values))
    print(len(max_probs))

    plt.figure(figsize=(10, 6))
    bins = np.linspace(min(ccp_values), max(ccp_values), 100)
    plt.hist(ccp_values, bins=bins, edgecolor='k', alpha=0.7)
    plt.title("Distribution of CCP Values")
    plt.xlabel("CCP")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    bins = np.linspace(min(max_probs), max(max_probs), 100)
    plt.hist(max_probs, bins=bins, edgecolor='k', alpha=0.7)
    plt.title("Distribution of Max Prob Values")
    plt.xlabel("max_prob")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def label_quantiles(json_data, all_info_seeking=False):
    def is_info_seeking(entry):
        if all_info_seeking:
            return entry.get("primary_tag", "").lower() == "information seeking"
        else:
            return (
                entry.get("primary_tag", "").lower() == "information seeking"
                and entry.get("other_tags") is None
            )
    ccp_values = []
    max_prob_values = []
    total_claims_list = []
    total_claim_count = []
    total_res_len = []

    for entry in json_data:
        count = 0
        claim_list = []  # Initialize list for this entry's claims

        if is_info_seeking(entry) and "claim_uncertainty" in entry:
            if entry['claim_uncertainty'] is None:
                continue

            for uncertainty in entry["claim_uncertainty"]:
                ccp_val = uncertainty.get("ccp")
                max_prob_val = uncertainty.get("max_prob")
                claims = uncertainty.get("claim")

                if claims is not None:
                    claim_list.append(claims)  # Store the claim text
                    count += 1

                if ccp_val is not None:
                    ccp_values.append(ccp_val)
                if max_prob_val is not None:
                    max_prob_values.append(max_prob_val)

            total_claim_count.append(count)  # Store total claims count per entry
            total_claims_list.append(claim_list)  # Store the list of claims per entry

        # import pdb; pdb.set_trace()
        # Calculate response length
        total_res_len.append(len(entry['response'].split()))
        # import pdb; pdb.set_trace()
    # Print data structure lengths for debugging
    print("Total entries processed:", len(total_claim_count))  # Number of processed entries
    print("Total claim lists stored:", len(total_claims_list))  # Number of lists of claims stored
    print("Total response lengths stored:", len(total_res_len))

    # Compute the average claim count
    average_claim_count = sum(total_claim_count) / len(total_claim_count) if total_claim_count else 0

    # Compute word count for all claims outside the loop
    all_claims = [claim for sublist in total_claims_list for claim in sublist]  # Flatten list
    word_counts = [len(claim.split()) for claim in all_claims]  # Compute word count per claim

    # Calculate average word count per claim
    average_word_count = sum(word_counts) / len(word_counts) if word_counts else 0

    # Calculate average response length
    average_res_len = sum(total_res_len) / len(total_res_len) if total_res_len else 0

    print("Average_claim_count:", average_claim_count)
    print("Average_word_count_in_claims:", average_word_count)
    print("Average_response_length:", average_res_len)

    # import pdb; pdb.set_trace()



    if len(ccp_values) == 0 or len(max_prob_values) == 0:
        print("No CCP or max_prob values found for info-seeking entries. Returning data unchanged.")
        return json_data

    ccp_series = pd.Series(ccp_values)
    max_prob_series = pd.Series(max_prob_values)

    quantiles = [0.75]
    ccp_quantiles = ccp_series.quantile(quantiles)
    max_prob_quantiles = max_prob_series.quantile(quantiles)

    print("\nQuantiles of CCP Values (info-seeking only):")
    print(ccp_quantiles)
    print("\nQuantiles of Max_Prob Values (info-seeking only):")
    print(max_prob_quantiles)

    for entry in json_data:
        if is_info_seeking(entry) and "claim_uncertainty" in entry:
            if entry['claim_uncertainty'] is None:
                continue
            for uncertainty in entry["claim_uncertainty"]:
                ccp_val = uncertainty.get("ccp")
                max_prob_val = uncertainty.get("max_prob")
                if ccp_val is not None:
                    for q, q_val in ccp_quantiles.items():
                        uncertainty[f'ccp_above_q{int(q*100)}'] = int(ccp_val > q_val)
                if max_prob_val is not None:
                    for q, q_val in max_prob_quantiles.items():
                        uncertainty[f'max_prob_above_q{int(q*100)}'] = int(max_prob_val > q_val)

    return json_data

def append_exceeding_claims_with_confidence(json_data, quantile_threshold, all_info_seeking=False):
    updated_data = copy.deepcopy(json_data)

    confident_count = 0
    info_seek_count = 0
    total_count = 0
    unconfident_count = 0
    total_exceeding_claims = 0

    for entry in updated_data:
        if all_info_seeking:
            is_info_seeking = (entry.get("primary_tag", "").lower() == "information seeking")
        else:
            is_info_seeking = (
                entry.get("primary_tag", "").lower() == "information seeking"
                and entry.get("other_tags") is None
            )
        if not is_info_seeking:
            continue

        info_seek_count += 1

        exceeding_claims = []
        if "claim_uncertainty" in entry and entry["claim_uncertainty"] is not None:
            exceeding_claims = [
                u["claim"] for u in entry["claim_uncertainty"]
                if quantile_threshold in u and u[quantile_threshold] == 1
            ]

        total_exceeding_claims += len(exceeding_claims)

        if exceeding_claims:
            if len(exceeding_claims) > 10:
                entry["surgery_response"] = (
                    entry.get("response", "")
                    + "\n\n<reflection>"
                    + "I am unconfident about the accuracy and the truthfulness of most of the information provided above."
                )
                unconfident_count += 1
            else:
                formatted_list = "\n".join(f"{i+1}. {fact}" for i, fact in enumerate(exceeding_claims))
                entry["surgery_response"] = (
                    entry.get("response", "")
                    + f"\n\n<reflection>\nThe following summarizes my uncertainty about some of the facts presented above:\n{formatted_list}"
                )
        else:
            entry["surgery_response"] = (
                entry.get("response", "")
                + "\n\n<reflection>I am confident about the accuracy and the truthfulness of the information provided."
            )
            confident_count += 1

        total_count += 1

    print("[append_exceeding_claims_with_confidence] Stats:")
    print("  - # entries with no exceeding claims (confident_count):", confident_count)
    print("  - # 'information seeking' entries encountered:", info_seek_count)
    print("  - # entries that triggered unconfident message:", unconfident_count)
    print("  - total # of entries processed in this loop:", total_count)

    if info_seek_count > 0:
        average_exceeding_claims = total_exceeding_claims / info_seek_count
    else:
        average_exceeding_claims = 0
    print("  - # of exceeding claims per 'information seeking' entry:", average_exceeding_claims)
    print()

    return updated_data

def transform_data(data, all_info_seeking=False, no_surgery=False):
    """
    Transforms the input data to include columns: chosen, rejected, prompt, and messages.

    - If no_surgery=True, the final assistant content is always the original response,
      and the system message is forced to SYS_MESSAGE_OTHER (no reflection).
    """
    transformed_data = []

    SYS_MESSAGE_INFO = (
        "You are a helpful assistant."
        "you should answer user's query first, providing a helpful and accurate response."
        "Then write a <reflection> section following your "
        "response, listing all the factual claims you made in your response that you are "
        "uncertain about.\n\n"
        "Output your reflection in the following format ONLY:\n"
        "<reflection>\n"
        "The following summarizes the facts that I am uncertain about in my answer:\n"
        "1. [factual claim 1 that you are uncertain about]\n"
        "2. [factual claim 2 that you are uncertain about]\n"
        "3. [factual claim 3 that you are uncertain about]\n"
        "...[more factual claims]..."
    )

    SYS_MESSAGE_OTHER = (
        "You are a helpful assistant."
        "you should answer user's query directly, providing a helpful and accurate response to the query."
    )

    for entry in data:
        # Determine if entry is "information seeking" (unless we override for NO_SURGERY)
        if all_info_seeking:
            is_info_seeking = (entry.get("primary_tag", "").lower() == "information seeking")
        else:
            is_info_seeking = (
                entry.get("primary_tag", "").lower() == "information seeking"
                and entry.get("other_tags") is None
            )

        instruction = entry.get("instruction", "")
        response = entry.get("response", "")
        surgery_response = entry.get("surgery_response", "")

        if no_surgery:
            # Always use direct "SYS_MESSAGE_OTHER"
            SYS_MESSAGE = SYS_MESSAGE_OTHER
            # Also, forcibly use the original response for assistant
            assistant_content = response
        else:
            # Normal logic
            if is_info_seeking:
                SYS_MESSAGE = SYS_MESSAGE_INFO
                assistant_content = surgery_response
            else:
                SYS_MESSAGE = SYS_MESSAGE_OTHER
                assistant_content = response

        # 2) Construct the multi-turn structure
        transformed_entry = {
            "prompt": instruction,
            "chosen": [
                {"content": SYS_MESSAGE, "role": "system"},
                {"content": instruction, "role": "user"},
                {"content": assistant_content, "role": "assistant"}
            ],
            "rejected": [
                {"content": SYS_MESSAGE, "role": "system"},
                {"content": instruction, "role": "user"},
                {"content": response, "role": "assistant"}
            ],
            "messages": [
                {"content": SYS_MESSAGE, "role": "system"},
                {"content": instruction, "role": "user"},
                {"content": assistant_content, "role": "assistant"}
            ],
            # Keep the original response for reference
            "response": response
        }

        transformed_data.append(transformed_entry)

    return transformed_data

def train_test_split(data, test_size=0.2, seed=42):
    random.seed(seed)
    # random.shuffle(data)
    split_index = int(len(data) * (1 - test_size))
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data

def main():
    print(f"Configuration -> DATANAME: {DATANAME}, CACHE_NAME: {CACHE_NAME}")
    print(f"ALL_INFO_SEEKING={ALL_INFO_SEEKING}, ONLY_OUTPUT_INFO_SEEK={ONLY_OUTPUT_INFO_SEEK}, NO_SURGERY={NO_SURGERY}")

    def is_info_seeking(entry):
        if ALL_INFO_SEEKING:
            return entry.get("primary_tag", "").lower() == "information seeking"
        else:
            return (
                entry.get("primary_tag", "").lower() == "information seeking"
                and entry.get("other_tags") is None
            )

    # 1. Load data
    data = load_jsonl(INPUT_FILE)
    print(f"Loaded {len(data)} entries from {INPUT_FILE}.")

    # 2. Remove entries with no response
    data, removed = remove_entries_with_none_response(data)
    print(f"Removed {len(removed)} entries because 'response' was None.")
    print(f"Now we have {len(data)} entries.\n")

    # 3. (Optional) Plot distributions
    if DO_PLOT:
        extract_and_plot_ccp(data)

    # 4. Label quantiles (harmless if we never actually use them)
    data = label_quantiles(data, all_info_seeking=ALL_INFO_SEEKING)

    # import pdb; pdb.set_trace()

    # 5. If NO_SURGERY is True, produce a single dataset with no reflection logic
    if NO_SURGERY:
        # We'll skip the entire quantile loop
        appended_data = data  # No changes to "surgery_response" needed

        if ONLY_OUTPUT_INFO_SEEK:
            appended_data = [ad for ad in appended_data if is_info_seeking(ad)]

        final_data = transform_data(
            appended_data,
            all_info_seeking=ALL_INFO_SEEKING,
            no_surgery=NO_SURGERY  # This forces system=SYS_MESSAGE_OTHER
        )

        # train_data, test_data = train_test_split(final_data, test_size=TEST_SIZE, seed=42)

        # Save a single file set
        train_path = os.path.join(OUTPUT_DIR, f"{CACHE_NAME}_train.jsonl")
        # test_path = os.path.join(OUTPUT_DIR, f"{CACHE_NAME}_test.jsonl")

        print(f"Length of train data: {len(train_data)}")
        # print(f"Length of test data: {len(test_data)}")

        save_jsonl(train_data, train_path)
        # save_jsonl(test_data, test_path)
        print(f"Saved processed data (no surgery) to:")
        print(f"  -> Train: {train_path}")
        # print(f"  -> Test:  {test_path}")

    else:
        # 6. Otherwise, do the multi-quantile pass
        quantile_suffixes = ["q50", "q65", "q75", "q85", "q95"]
        processed_dict = {}

        for q in quantile_suffixes:
            threshold_key = f"ccp_above_{q}"
            print(f"Processing data with threshold: {threshold_key}")

            appended_data = append_exceeding_claims_with_confidence(
                data, threshold_key, all_info_seeking=ALL_INFO_SEEKING
            )

            if ONLY_OUTPUT_INFO_SEEK:
                appended_data = [ad for ad in appended_data if is_info_seeking(ad)]

            final_data = transform_data(
                appended_data,
                all_info_seeking=ALL_INFO_SEEKING,
                no_surgery=NO_SURGERY
            )

            # train_data, test_data = train_test_split(final_data, test_size=TEST_SIZE, seed=42)
            processed_dict[q] = {"train": train_data, "test": test_data}

        # 7. Save multi-quantile results
        for q, subsets in processed_dict.items():
            train_path = os.path.join(OUTPUT_DIR, f"{CACHE_NAME}_surgery_{q}_train.jsonl")
            # test_path = os.path.join(OUTPUT_DIR, f"{CACHE_NAME}_surgery_{q}_test.jsonl")

            print(f"Length of train data: {len(subsets['train'])}")
            # print(f"Length of test data: {len(subsets['test'])}")

            save_jsonl(subsets["train"], train_path)
            # save_jsonl(subsets["test"], test_path)

            print(f"Saved processed data for threshold {q}:")
            print(f"  -> Train: {train_path}")
            # print(f"  -> Test:  {test_path}")


if __name__ == "__main__":
    main()
