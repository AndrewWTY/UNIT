import argparse
import copy
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 1. CLI PARSER
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Process and split classification JSONL data"
    )
    p.add_argument("--input_file", required=True, help="Path to the input JSONL file")
    p.add_argument("--output_dir", default="", help="Directory for outputs")
    p.add_argument(
        "--test_size",
        type=float,
        default=0.0,
        help="Fraction of data reserved for a test split",
    )
    p.add_argument(
        "--do_plot", action="store_true", help="Plot distributions of CCP and max_prob"
    )
    p.add_argument(
        "--all_info",
        action="store_true",
        help="Treat *all* primary_tag=='information seeking' rows as info-seeking "
        "(ignore other_tags)",
    )
    p.add_argument(
        "--only_info",
        action="store_true",
        help="After processing, keep only info-seeking rows in the final output",
    )
    p.add_argument(
        "--no_surgery",
        action="store_true",
        help="Disable surgery / reflection logic (outputs plain responses)",
    )
    return p.parse_args()


args = parse_args()
INPUT_FILE = args.input_file
OUTPUT_DIR = args.output_dir or "."
TEST_SIZE = args.test_size
DO_PLOT = args.do_plot
ALL_INFO_SEEKING = args.all_info
ONLY_OUTPUT_INFO_SEEK = args.only_info
NO_SURGERY = args.no_surgery


os.makedirs(OUTPUT_DIR, exist_ok=True)
CACHE_NAME = os.path.splitext(os.path.basename(INPUT_FILE))[0]
if ONLY_OUTPUT_INFO_SEEK:
    CACHE_NAME += "_only_info_seek"
if NO_SURGERY:
    CACHE_NAME += "_no_surgery"

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            try:
                data.append(json.loads(raw))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipped malformed line: {e}")
    return data


def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def remove_entries_with_none_response(data):
    new, dropped = [], 0
    for d in data:
        if d.get("response") is None:
            dropped += 1
        else:
            new.append(d)
    return new, dropped


def extract_and_plot_ccp(data):
    ccp, mp = [], []
    for d in data:
        for u in d.get("claim_uncertainty") or []:
            if (v := u.get("ccp")) is not None:
                ccp.append(v)
            if (v := u.get("max_prob")) is not None:
                mp.append(v)

    if not ccp or not mp:
        print("[WARN] No CCP / max_prob numbers found – skipping plot.")
        return

    ccp_series, mp_series = pd.Series(ccp), pd.Series(mp)
    quantiles = [0.50, 0.65, 0.75, 0.85, 0.95]
    print("CCP summary:\n", ccp_series.describe(), "\n")
    print("max_prob summary:\n", mp_series.describe(), "\n")
    print("CCP quantiles:\n", ccp_series.quantile(quantiles), "\n")
    print("max_prob quantiles:\n", mp_series.quantile(quantiles), "\n")

    # CCP histogram
    plt.figure(figsize=(10, 5))
    bins = np.linspace(min(ccp), max(ccp), 100)
    plt.hist(ccp, bins=bins, edgecolor="k", alpha=0.7)
    plt.title("Distribution of CCP values")
    plt.xlabel("ccp")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    # max_prob histogram
    plt.figure(figsize=(10, 5))
    bins = np.linspace(min(mp), max(mp), 100)
    plt.hist(mp, bins=bins, edgecolor="k", alpha=0.7)
    plt.title("Distribution of max_prob values")
    plt.xlabel("max_prob")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def label_quantiles(data, all_info_seeking=False):
    """Add booleans (ccp_above_qXX / max_prob_above_qXX) to each uncertainty block.
    Returns (new_data, ccp_q75)."""
    def is_info(d):
        if all_info_seeking:
            return d.get("primary_tag", "").lower() == "information seeking"
        return d.get("primary_tag", "").lower() == "information seeking" and d.get(
            "other_tags"
        ) is None

    # Gather stats only from info-seeking rows
    ccp_vals, mp_vals = [], []
    for d in data:
        if not is_info(d):
            continue
        for u in d.get("claim_uncertainty") or []:
            if (v := u.get("ccp")) is not None:
                ccp_vals.append(v)
            if (v := u.get("max_prob")) is not None:
                mp_vals.append(v)

    if not ccp_vals:  
        return data, None

    ccp_series, mp_series = pd.Series(ccp_vals), pd.Series(mp_vals)
    quants = [0.75]
    ccp_q = ccp_series.quantile(quants)
    mp_q = mp_series.quantile(quants)

    # Annotate in place
    for d in data:
        if not is_info(d):
            continue
        for u in d.get("claim_uncertainty") or []:
            ccp_val, mp_val = u.get("ccp"), u.get("max_prob")
            if ccp_val is not None:
                for q, qv in ccp_q.items():
                    u[f"ccp_above_q{int(q*100)}"] = int(ccp_val > qv)
            if mp_val is not None:
                for q, qv in mp_q.items():
                    u[f"max_prob_above_q{int(q*100)}"] = int(mp_val > qv)

    return data, ccp_q[0.75]


def add_certainty_fields(data, ccp_q75, all_info_seeking=False):
    """Adds 'info-seeking', 'certain_claims', 'uncertain_claims' columns."""
    def is_info(d):
        if all_info_seeking:
            return d.get("primary_tag", "").lower() == "information seeking"
        return d.get("primary_tag", "").lower() == "information seeking" and d.get(
            "other_tags"
        ) is None

    for idx, d in enumerate(data):
        d["info-seeking"] = is_info(d)

        certain, uncertain = [], []
        for u in d.get("claim_uncertainty") or []:
            claim, ccp_val = u.get("claim"), u.get("ccp")
            if claim is None or ccp_val is None:
                continue
            (uncertain if ccp_val > ccp_q75 else certain).append(claim)

        d["certain_claims"] = certain
        d["uncertain_claims"] = uncertain
    return data


def append_exceeding_claims_with_confidence(data, threshold_key, all_info_seeking=False):
    """Attach a 'surgery_response' that includes reflection text."""
    out = copy.deepcopy(data)

    def is_info(d):
        if all_info_seeking:
            return d.get("primary_tag", "").lower() == "information seeking"
        return d.get("primary_tag", "").lower() == "information seeking" and d.get(
            "other_tags"
        ) is None

    for d in out:
        if not is_info(d):
            continue

        exceeding = [
            u["claim"]
            for u in d.get("claim_uncertainty") or []
            if u.get(threshold_key) == 1
        ]

        if not exceeding:
            d["surgery_response"] = (
                d["response"]
                + "\n\n<reflection>I am confident about the accuracy and truthfulness of the information provided."
            )
        elif len(exceeding) > 10:
            d["surgery_response"] = (
                d["response"]
                + "\n\n<reflection>I am unconfident about the accuracy and truthfulness of most of the information provided above."
            )
        else:
            bullets = "\n".join(f"{i+1}. {c}" for i, c in enumerate(exceeding))
            d["surgery_response"] = (
                d["response"]
                + "\n\n<reflection>\nThe following summarizes my uncertainty about some of the facts presented above:\n"
                + bullets
            )
    return out


# ---------- Final multi-turn structure ----------------------------------- #
def transform_data(rows, all_info_seeking=False, no_surgery=False):
    out = []

    SYS_MESSAGE_INFO = (
        "You are a helpful assistant. "
        "You should answer the user's query first. "
        "Then write a <reflection> section listing the factual claims you are uncertain about."
    )
    SYS_MESSAGE_OTHER = (
        "You are a helpful assistant. "
        "Answer the user's query directly and accurately."
    )

    def is_info(d):
        if all_info_seeking:
            return d.get("primary_tag", "").lower() == "information seeking"
        return d.get("primary_tag", "").lower() == "information seeking" and d.get(
            "other_tags"
        ) is None

    for d in rows:
        # choose system + assistant content
        if no_surgery:
            sys_msg = SYS_MESSAGE_OTHER
            assistant_content = d["response"]
        else:
            if is_info(d):
                sys_msg = SYS_MESSAGE_INFO
                assistant_content = d.get("surgery_response", d["response"])
            else:
                sys_msg = SYS_MESSAGE_OTHER
                assistant_content = d["response"]

        transformed = {
            "prompt": d.get("instruction", ""),
            "chosen": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": d.get("instruction", "")},
                {"role": "assistant", "content": assistant_content},
            ],
            "rejected": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": d.get("instruction", "")},
                {"role": "assistant", "content": d["response"]},
            ],
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": d.get("instruction", "")},
                {"role": "assistant", "content": assistant_content},
            ],
            # carry-through extras
            "response": d["response"],
            "info-seeking": d.get("info-seeking", False),
            "certain_claims": d.get("certain_claims", []),
            "uncertain_claims": d.get("uncertain_claims", []),
        }
        out.append(transformed)
    return out


def train_test_split(rows, test_size=0.0, seed=42):
    if test_size <= 0.0:
        return rows, []
    random.seed(seed)
    random.shuffle(rows)
    split = int(len(rows) * (1 - test_size))
    return rows[:split], rows[split:]


def main():
    print(f"[INFO] Loading {INPUT_FILE}")
    data = load_jsonl(INPUT_FILE)
    print(f"[INFO] Loaded {len(data)} records")

    # remove None responses
    data, dropped = remove_entries_with_none_response(data)
    if dropped:
        print(f"[INFO] Dropped {dropped} rows with null response")

    if DO_PLOT:
        extract_and_plot_ccp(data)

    # quantile labels
    data, ccp_q75 = label_quantiles(data, ALL_INFO_SEEKING)

    # certainty fields
    if ccp_q75 is not None:
        data = add_certainty_fields(data, ccp_q75, ALL_INFO_SEEKING)

    # path helpers
    def out_path(name):
        return os.path.join(OUTPUT_DIR, name)

    if NO_SURGERY:
        if ONLY_OUTPUT_INFO_SEEK:
            data = [d for d in data if d.get("info-seeking")]
        final = transform_data(data, ALL_INFO_SEEKING, NO_SURGERY)
        train, test = train_test_split(final, TEST_SIZE)

        save_jsonl(train, out_path(f"{CACHE_NAME}_train.jsonl"))
        if test:
            save_jsonl(test, out_path(f"{CACHE_NAME}_test.jsonl"))
        print(f"[DONE] Saved {len(train)} train (+{len(test)} test) rows")
        return

    quantile_suffixes = ["q75"]
    for q in quantile_suffixes:
        key = f"ccp_above_{q}"
        print(f"[INFO] Generating dataset for threshold {key}")

        augmented = append_exceeding_claims_with_confidence(
            data, key, ALL_INFO_SEEKING
        )
        if ONLY_OUTPUT_INFO_SEEK:
            augmented = [d for d in augmented if d.get("info-seeking")]

        final = transform_data(augmented, ALL_INFO_SEEKING, NO_SURGERY)
        train, test = train_test_split(final, TEST_SIZE)

        save_jsonl(train, out_path(f"{CACHE_NAME}_surgery_{q}_train.jsonl"))
        if test:
            save_jsonl(test, out_path(f"{CACHE_NAME}_surgery_{q}_test.jsonl"))
        print(
            f"    ↳ saved {len(train)} train (+{len(test)} test) rows for q={q}"
        )


if __name__ == "__main__":
    main()
