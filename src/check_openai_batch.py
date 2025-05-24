import os
import openai
import argparse
import pandas as pd
import json
from pprint import pprint
from openai import OpenAI
import time  # Added import for time module

os.environ['OPENAI_API_KEY'] = "replace_with_your_openai_api_key"  # Set your OpenAI API key here
openai.api_key = os.getenv("OPENAI_API_KEY")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--cache_name", type=str, default='test_cache')
    parser.add_argument("--match", default=False, action='store_true')
    parser.add_argument("--extract", default=False, action='store_true')
    parser.add_argument("--final", default=False, action='store_true')
    args = parser.parse_args()

    client = OpenAI()

    sub_cache_flags = [args.match, args.extract, args.final]
    if sum(sub_cache_flags) != 1:
        raise ValueError("Exactly one of --match, --extract, or --final must be specified.")


    if args.match:
        sub_cache_name = 'match'
    elif args.extract:
        sub_cache_name = 'extract'
    else:
        raise NotImplementedError("Final flag is not yet implemented.")


    if args.id is None and args.input_file is not None:
        txt_filename = f"{args.cache_name}_{sub_cache_name}.txt"
        batch_id_file = os.path.join('../batch_id_database', txt_filename)
        if os.path.exists(batch_id_file):
            with open(batch_id_file, 'r') as f:
                batch_ids = [line.strip() for line in f if line.strip()]
            if not batch_ids:
                print(f"No batch IDs found in '{batch_id_file}'.")
                exit(1)
            print(f"Batch IDs {batch_ids} retrieved from '{batch_id_file}'.")
        else:
            print(f"Batch ID file '{batch_id_file}' does not exist.")
            exit(1)
    else:
        batch_ids = [args.id]

    output_json = {}

    pending_batch_ids = set(batch_ids)

    while pending_batch_ids:
        completed_this_round = []
        for b_id in pending_batch_ids:
            status = client.batches.retrieve(b_id)
            if status.status == 'completed':
                completed_this_round.append(b_id)
                input_file_content = client.files.content(status.input_file_id)
                file_response = client.files.content(status.output_file_id)
                for idx, item in enumerate(file_response.text.strip("\n").split('\n')):
                    item = json.loads(item)
                    custom_id = item['custom_id']
                    response = item['response']['body']['choices'][0]['message']['content']
                    output_json[custom_id] = response
            else:
                print(f"Batch {b_id} status: {status.status}. Still waiting...")

        for completed_id in completed_this_round:
            pending_batch_ids.remove(completed_id)

        if pending_batch_ids:
            time.sleep(60*5) 

    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(output_json, f)
        print(f"All completed batches output saved to '{args.output_file}'.")
    else:
        pprint(output_json)





