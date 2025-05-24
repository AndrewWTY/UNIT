#!/usr/bin/env python
import argparse
import asyncio
import json
import os
import pickle
import random
import time

import pandas as pd
import tiktoken
from openai import AsyncOpenAI

# --------------------------------------------------
# Environment / model constants
# --------------------------------------------------
os.environ['OPENAI_API_KEY'] = 'your_openai_api_key_here'   # <-- put your key here
MODEL_DICT = {
    'gpt-4o-mini': 'gpt-4o-mini',
    'gpt-4o': 'gpt-4o',
}
INPUT_COST_DICT  = {'gpt-4o-mini': 0.15, 'gpt-4o': 2.5}
OUTPUT_COST_DICT = {'gpt-4o-mini': 0.60, 'gpt-4o': 10.0}

SYSTEM = """You are a helpful AI assistant."""
PROMPT_TMPL = """# Instruction
Please label the task tags for the user query.
## User Query
'''{input}'''
## Tagging the user input
Please label the task tags for the user query. You will need to analyze the user query and select the most relevant task tag from the list below.
all_task_tags = [
"Information seeking", # Users ask for specific information or facts about various topics.
"Reasoning", # Queries require logical thinking, problem−solving, or processing of complex ideas.
"Planning", # Users need assistance in creating plans or strategies for activities and projects.
"Editing", # Involves editing, rephrasing, proofreading, or other tasks related to the composition of general written content.
"Coding & Debugging", # Users seek help with writing, reviewing, or fixing code in programming.
"Math", # Queries related to mathematical concepts, problems, and calculations.
"Role playing", # Users engage in scenarios requiring ChatGPT to adopt a character or persona.
"Data analysis", # Requests involve interpreting data, statistics, or performing analytical tasks.
"Creative writing", # Users seek assistance with crafting stories, poems, or other creative texts.
"Advice seeking", # Users ask for recommendations or guidance on various personal or professional issues.
"Brainstorming", # Involves generating ideas, creative thinking, or exploring possibilities.
"Others" # Any queries that do not fit into the above categories or are of a miscellaneous nature.
]
## Output Format:
Note that you can only select a single primary tag. Other applicable tags can be added to the list of other tags.
Now, please output your tags below in a json format by filling in the placeholders in <...>:
'''
{{
"primary_tag": "<primary tag>",
"other_tags": ["<tag 1>", "<tag 2>", ... ]
}}
'''"""

# --------------------------------------------------
# Utilities
# --------------------------------------------------
def batchify(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

async def achat(client, model, msg, seed=42, temperature=0):
    out = await client.chat.completions.create(
        messages=msg, model=model, seed=seed, temperature=temperature
    )
    return out.choices[0].message.content

async def create_answers_async(client, model, messages, batch_size=20,
                               seed=42, temperature=0, cache_name="batch"):
    """Send prompts in parallel batches (with simple on-disk caching)."""
    all_answers = []
    for bi, batch in enumerate(batchify(messages, batch_size)):
        cache_file = f"{cache_name}{bi}.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as fh:
                answers = pickle.load(fh)
            print(f"Batch {bi} loaded from cache.")
        else:
            try:
                answers = await asyncio.gather(
                    *[achat(client, model, m, seed, temperature) for m in batch]
                )
                with open(cache_file, "wb") as fh:
                    pickle.dump(answers, fh)
                print(f"Batch {bi} completed via API.")
            except Exception as e:
                print(f"Batch {bi} failed: {e}")
                answers = []
        all_answers.extend(answers)
        time.sleep(1)
    return all_answers

def token_price(txt, model, price_dict):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(txt)) / 1_000_000 * price_dict[model]

# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction_file", required=True,
                        help="Path to .jsonl file with at least an 'instruction' column")
    parser.add_argument("--output_file", required=True,
                        help="Destination .xlsx file")
    parser.add_argument("--llm", default="gpt-4o", choices=list(MODEL_DICT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=-1,
                        help="Randomly sample n rows for quick runs")
    parser.add_argument("--batch", type=int, default=10)
    args = parser.parse_args()

    random.seed(args.seed)

    # ---------- load jsonl ----------
    rows = [json.loads(l) for l in open(args.instruction_file, "r", encoding="utf-8")]
    if args.sample > 0:
        rows = random.sample(rows, args.sample)

    df = pd.DataFrame(rows)                         # keep *all* original columns
    if "instruction" not in df.columns:
        raise ValueError("'instruction_file' must contain an 'instruction' field on every line")

    # ---------- build prompts ----------
    prompts, prompt_messages, input_prices = [], [], []
    for instr in df["instruction"]:
        p = PROMPT_TMPL.format(input=instr)
        prompts.append(p)

        msg = [{"role": "system", "content": SYSTEM},
               {"role": "user",   "content": p}]
        prompt_messages.append(msg)

        input_prices.append(token_price(SYSTEM + p, MODEL_DICT[args.llm], INPUT_COST_DICT))

    print(f"Estimated input cost: ${sum(input_prices):.4f}")

    # ---------- call LLM ----------
    client = AsyncOpenAI()
    responses = asyncio.run(
        create_answers_async(client, MODEL_DICT[args.llm], prompt_messages,
                             batch_size=args.batch, seed=args.seed)
    )

    # ---------- parse tag json ----------
    primary, others, output_prices = [], [], []
    for r in responses:
        try:
            tag_obj = json.loads(r)
            primary.append(tag_obj.get("primary_tag", ""))
            others.append(tag_obj.get("other_tags", []))
        except json.JSONDecodeError:
            primary.append("")
            others.append([])
        output_prices.append(token_price(r, MODEL_DICT[args.llm], OUTPUT_COST_DICT))

    # ---------- assemble & save ----------
    df["primary_tag"] = primary
    df["other_tags"]  = others

    # df.to_excel(args.output_file, index=False)
    df.to_json(args.output_file, orient="records", lines=True)
    print(f"Saved {len(df)} rows to {args.output_file}")
