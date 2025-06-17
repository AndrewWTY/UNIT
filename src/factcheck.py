import re
import string
import time
import diskcache as dc
from diskcache import FanoutCache
import asyncio
import openai
from openai import OpenAI, AsyncOpenAI
import tiktoken
import json
import random
import pandas as pd
import os
import argparse
from wild_retrieval import WildRetrieval
from factscore_retrieval import DocDB, Retrieval
from tqdm import tqdm
from datasets import load_dataset
import pickle


MODEL_DICT = {
    'gpt-4o-mini': 'gpt-4o-mini',
    'gpt-4o': 'gpt-4o',
}
INPUT_COST_DICT = {
    'gpt-4o-mini': 0.15,
    'gpt-4o': 2.5,
}
OUTPUT_COST_DICT = {
    'gpt-4o-mini': 0.6,
    'gpt-4o': 10,
}


SYSTEM_PROMPT = "You are a helpful AI assistant."
# DEFINITION = "Answer the question about {topic} based on the following references and your knowledge.\n\n"
# PROMPT = "\n\nInput: {} True or False?\nOutput:"
USER_PROMPT = """Analyze the following question and its associated claim:

Question: {input}

Claim: {claim}

Some context that might be helpful to fact-check the Claim:
{context}

Now answer: is all information provided in the <claim> true given the context and your latest knowledge?
"""

SUMMARY_PROMPT = """Question: {input}

Claim: {claim}

Is the above claim true?

Reply: {reply}

Summarize this reply into one word, whether the claim is true: "True", "False" or "Not known".
"""


def get_input_price(input_str, model):
    encoding = tiktoken.encoding_for_model(MODEL_DICT[model])
    input_len = len(encoding.encode(input_str))
    input_cost = input_len / 1000000 * INPUT_COST_DICT[model]
    return input_cost


def get_output_price(output_str, model):
    encoding = tiktoken.encoding_for_model(MODEL_DICT[model])
    output_len = len(encoding.encode(output_str))
    output_cost = output_len / 1000000 * OUTPUT_COST_DICT[model]
    return output_cost


class OpenAIChat:
    """
    Allows for the implementation of a singleton class to chat with OpenAI model for dataset marking.
    """

    def __init__(
        self,
        openai_model: str = "gpt-4o-mini",
        cache_path: str = "openai_cache",
        cache_name: str = "cache",
        system_prompt: str = SYSTEM_PROMPT,
    ):
        """
        Parameters
        ----------
        openai_model: str
            the model to use in OpenAI to chat.
        """
        self.sys_prompts = system_prompt
        api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is not None:
            openai.api_key = api_key
        self.openai_model = openai_model

        self.cache_path = os.path.join(cache_path, f"{cache_name}.diskcache")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

    def ask(self, message: str) -> str:
        cache_settings = dc.DEFAULT_SETTINGS.copy()
        cache_settings["eviction_policy"] = "none"
        cache_settings["size_limit"] = int(1e12)
        cache_settings["cull_limit"] = 0
        with dc.Cache(self.cache_path, **cache_settings) as openai_responses:
            if (self.openai_model, message) in openai_responses:
                reply = openai_responses[(self.openai_model, message)]
                print("Loaded from cache")
            else:
                if openai.api_key is None:
                    raise Exception(
                        "Cannot ask OpenAI without token. "
                        "Please specify OPENAI_API_KEY in environment parameters."
                    )
                messages = [
                    {"role": "system", "content": self.sys_prompts},
                    {"role": "user", "content": message},
                ]
                chat = self._send_request(messages)
                reply = chat.choices[0].message.content

                openai_responses[(self.openai_model, message)] = reply

        if "please provide" in reply.lower():
            return ""
        if "to assist you" in reply.lower():
            return ""
        if "as an ai language model" in reply.lower():
            return ""

        return reply

    async def async_ask(self, message_list):
        if openai.api_key is None:
            raise Exception(
                "Cannot ask OpenAI without token. "
                "Please specify OPENAI_API_KEY in environment parameters."
            )
        cache_settings = dc.DEFAULT_SETTINGS.copy()
        cache_settings["eviction_policy"] = "none"
        cache_settings["size_limit"] = int(1e12)
        cache_settings["cull_limit"] = 0
        replies = {}
        async_client = AsyncOpenAI()
        with dc.Cache(self.cache_path, **cache_settings) as openai_responses:
            messages_to_run = []
            indices_to_get = []
            for i, message in enumerate(message_list):
                if (self.openai_model, message) in openai_responses:
                    reply = openai_responses[(self.openai_model, message)]
                    if reply != "":
                        replies[i] = reply
                        print("Loaded from cache")
                    else:
                        indices_to_get.append(i)
                        messages_to_run.append(message)
                else:
                    indices_to_get.append(i)
                    messages_to_run.append(message)

            formatted_messages = [[
                {"role": "system", "content": self.sys_prompts},
                {"role": "user", "content": m},
            ] for m in messages_to_run]
            sleep_time_values = (5, 10, 30, 60, 120)
            for i in range(len(sleep_time_values)):
                try:
                    answers = await asyncio.gather(*[self._async_send_request(async_client, me) for me in formatted_messages])
                    for idx, answer, message in zip(indices_to_get, answers, messages_to_run):
                        replies[idx] = answer
                        openai_responses[(self.openai_model, message)] = answer
                    assert(len(message_list) == len(replies))
                    reply_list = [replies[j] for j in range(len(message_list))]
                    return reply_list
                except Exception as e:
                    sleep_time = sleep_time_values[i]
                    print(
                        f"Request to OpenAI failed with exception: {e}. Retry #{i}/5 after {sleep_time} seconds."
                    )
                    await asyncio.sleep(sleep_time)

        for idx in indices_to_get:
            replies[idx] = ""
        reply_list = [replies[j] for j in range(len(message_list))]
        return reply_list

    async def _async_send_request(self, client, message):
        out = await client.chat.completions.create(
            model=self.openai_model,
            messages=message,
            temperature=0,  # for deterministic outputs
            max_tokens=1024,
        )
        return out.choices[0].message.content

    def _send_request(self, messages):
        sleep_time_values = (5, 10, 30, 60, 120)
        for i in range(len(sleep_time_values)):
            try:
                return openai.OpenAI().chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    temperature=0,  # for deterministic outputs
                    max_tokens=1024,
                )
            except Exception as e:
                sleep_time = sleep_time_values[i]
                print(
                    f"Request to OpenAI failed with exception: {e}. Retry #{i}/5 after {sleep_time} seconds."
                )
                time.sleep(sleep_time)

        return openai.OpenAI().chat.completions.create(
            model=self.openai_model,
            messages=messages,
            temperature=0,  # for deterministic outputs
            max_tokens=1024,
        )


def batchify(lst, batch_size):
    """Split the list `lst` into sublists of size `batch_size`."""
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


def ask_open_ai(args, prompts, batch_result, sub_cache_name, system_prompt=SYSTEM_PROMPT, real_time=False):
    input_prices = []
    parsed_prompts = []
    for p in prompts:
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": p},
        ]
        parsed_prompts.append(message)
        input_prices.append(get_input_price(p, args.llm))

    responses = []
    # Use batch API
    if batch_result is None:
        if real_time:
            print(f"Input cost with Real Time", sum(input_prices))
            openai_chat = OpenAIChat(system_prompt=system_prompt, openai_model=MODEL_DICT[args.llm], cache_name=f'{args.cache_name}_{sub_cache_name}')
            prompt_batches = batchify(prompts, batch_size=10)
            for p_batch in tqdm(prompt_batches, total=len(prompt_batches), desc='OpenAI Async Chat'):
                response_batch = asyncio.run(openai_chat.async_ask(p_batch))
                responses.extend(response_batch)
            # for p in tqdm(prompts, total=len(prompts), desc='OpenAI Chat'):
            #     responses.append(openai_chat.ask(p))
        else:
            print(f"Input cost with Batch API", sum(input_prices) / 2)
            # === Edited Code Starts Here ===
            # Instead of creating a single batch, we break down the prompts into chunks if they exceed 50,000
            MAX_BATCH_SIZE = 50000
            total_prompts = len(parsed_prompts)
            # We will process in chunks
            batch_ids = []
            for chunk_start in range(0, total_prompts, MAX_BATCH_SIZE):
                chunk_end = min(chunk_start + MAX_BATCH_SIZE, total_prompts)
                chunk_prompts = parsed_prompts[chunk_start:chunk_end]

                batched_prompts = []
                for idx, p in enumerate(chunk_prompts):
                    custom_id = args.cache_name + f'_{sub_cache_name}_' + str(chunk_start + idx)
                    batched_prompts.append({
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": MODEL_DICT[args.llm],
                            "messages": p,
                            "temperature": 0,
                        }
                    })

                # Write batch prompts to file
                os.makedirs('evaluate_database', exist_ok=True)
                batch_prompt_filename = 'evaluate_database/' + args.cache_name + f'_{sub_cache_name}_batch_prompt_{chunk_start}_{chunk_end}.jsonl'
                with open(batch_prompt_filename, 'w') as f:
                    for bp in batched_prompts:
                        f.write(json.dumps(bp))
                        f.write('\n')

                client = OpenAI()
                # Create input file for batch
                batch_input_file = client.files.create(
                    file=open(batch_prompt_filename, "rb"),
                    purpose="batch"
                )
                print("Input file object:", batch_input_file)
                batch_input_file_id = batch_input_file.id
                batch_obj = client.batches.create(
                    input_file_id=batch_input_file_id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                    metadata={
                        "description": args.cache_name + f'_{sub_cache_name}_batch_prompt_{chunk_start}_{chunk_end}'
                    }
                )
                print("Batch Object:", batch_obj)
                batch_id = batch_obj.id
                # Append the batch_id to the list
                batch_ids.append(batch_id)

            # After the loop ends, write all batch_ids to a single file
            txt_filename = f"{args.cache_name}_{sub_cache_name}.txt"
            os.makedirs('batch_id_database', exist_ok=True)
            output_path = os.path.join('batch_id_database', txt_filename)
            with open(output_path, 'w') as f:
                for b_id in batch_ids:
                    f.write(b_id + "\n")
            # After submitting all batches, exit since we are not processing responses in real time
            exit(0)
            # === Edited Code Ends Here ===
    else:
        with open(batch_result, 'r') as f:
            response_dict = json.load(f)
        for idx in range(len(parsed_prompts)):
            responses.append(response_dict[args.cache_name + f'_{sub_cache_name}_' + str(idx)])
    output_costs = 0
    for r in responses:
        output_costs += get_output_price(r, args.llm)
    if real_time:
        print(f"Output cost with Batch API", output_costs)
    else:
        print(f"Output cost with Batch API", output_costs / 2)
    return responses


async def achat(client, model, message, seed=42, temperature=0):
    out = await client.chat.completions.create(
        messages=message,
        model=model,
        seed=seed,
        temperature=temperature,
    )
    return out.choices[0].message.content


async def create_answers_async(model, messages, cache_path, batch_size=5, seed=42, temperature=0):
    # async answering
    client = AsyncOpenAI()
    batched_msgs = batchify(messages, batch_size)
    print("{} batches to run.".format(len(batched_msgs)))
    all_answers = []
    cache_settings = dc.DEFAULT_SETTINGS.copy()
    cache_settings["eviction_policy"] = "none"
    cache_settings["size_limit"] = int(1e12)
    cache_settings["cull_limit"] = 0
    error_batches = []
    with dc.Cache(cache_path, **cache_settings) as litellm_responses:
        for i, batch in tqdm(enumerate(batched_msgs), total=len(batched_msgs)):
            mapping_list = []
            cache_miss_msgs = []
            cache_hit_responses = []
            for msg_in_batch in batch:
                if (model, msg_in_batch) in litellm_responses:
                    mapping_list.append(len(cache_hit_responses) + 1)
                    cache_hit_responses.append(litellm_responses[(model, msg_in_batch)]['response'])
                else:
                    mapping_list.append(- len(cache_miss_msgs) - 1)
                    cache_miss_msgs.append(msg_in_batch)

            if len(cache_miss_msgs) == 0:
                all_answers.extend(cache_hit_responses)
                print(f"Batch {i} entirely Loaded")
            else:
                try:
                    answers = await asyncio.gather(*[achat(client, model, m, seed, temperature) for m in cache_miss_msgs])
                    for msg, res in zip(cache_miss_msgs, answers):
                        litellm_responses[(model, msg)] = {'response': res}
                    merged_responses = []
                    for idx in mapping_list:
                        if idx > 0:
                            merged_responses.append(cache_hit_responses[idx - 1])
                        else:
                            merged_responses.append(answers[- idx - 1])
                    all_answers.extend(merged_responses)
                    print(f"Batch {i} Done")
                except Exception as e:
                    print(f"Batch {i} Error while gathering answers: {e}")
                    error_batches.append(i)

    return all_answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='')
    parser.add_argument("--prompt_file", type=str, default='')
    parser.add_argument("--llm", type=str, default="gpt-4o-mini", choices=['gpt-4o-mini', 'gpt-4o'])
    parser.add_argument("--llm_batch_size", type=int, default=20)
    parser.add_argument("--retriever", type=str, default="gtr", choices=['gtr', 'bm25'])
    parser.add_argument("--db", type=str, default="enwiki", choices=['enwiki', 'wildhalu'])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--cache_name", type=str, default='', help='Cache name for OpenAI batch API')
    parser.add_argument("--batch_result_fc", type=str, default=None)
    parser.add_argument("--batch_result_sum", type=str, default=None)
    parser.add_argument("--fc_real_time", default=False, action='store_true')
    parser.add_argument("--sum_real_time", default=False, action='store_true')
    parser.add_argument("--output_file", type=str, default='')
    parser.add_argument("--sample", type=int, default=-1)
    args = parser.parse_args()

    if args.db == 'wildhalu':
        wildhalu_ds = load_dataset("wentingzhao/WildHallucinations", split="train")
    data_objs = []
    with open(args.input_file, 'r') as f:
        for line in f.readlines():
            obj = json.loads(line)
            obj['unreflected_claim_fc'] = []
            obj['unreflected_claim_sum'] = []
            obj['reflected_claim_fc'] = []
            obj['reflected_claim_sum'] = []
            data_objs.append(obj)

    if args.sample > 0:
        data_objs = data_objs[:args.sample]

    if not os.path.exists(args.prompt_file):
        if 'gtr' in args.retriever:
            retriever_type = 'gtr-t5-large'
        else:
            retriever_type = 'bm25'

        if args.db == 'enwiki':
            db_path = 'factcheck_cache/enwiki-20230401.db'
            data_path = ''
            cache_path = f'factcheck_cache/retrieval-enwiki-20230401-{args.retriever}.json'
            embed_cache_path = f'factcheck_cache/retrieval-enwiki-20230401-{args.retriever}.pkl'
            batch_size = 256
            database = DocDB(db_path=db_path, data_path=data_path)
            retriever = Retrieval(database, cache_path, embed_cache_path, batch_size=batch_size, retrieval_type=retriever_type)
        else:
            cache_path = f'factcheck_cache/retrieval-wildhalu-{args.retriever}.json'
            embed_cache_path = f'factcheck_cache/retrieval-wildhalu-{args.retriever}.pkl'
            retriever = WildRetrieval(batch_size=256, cache_path=cache_path, embed_cache_path=embed_cache_path,
                                      retrieval_type=retriever_type)

        claims_idx = []
        prompts = []
        inputs = []
        claims = []
        for idx, item in tqdm(enumerate(data_objs), total=len(data_objs), desc="Retrieve Passages"):
            if item['unreflected_answer_claim'] is None or item['reflected_answer_claim'] is None:
                continue
            entity = item['entity']
            if args.db == 'wildhalu':
                entity = entity[:-1] if entity.endswith('?') else entity
                if entity not in wildhalu_ds['entity']:
                    continue
            else:
                try:
                    retriever.get_passages(entity, "", args.top_k)
                except Exception as e:
                    print(str(e))
                    print(f"=== Entity {entity} does not exist in database ====")
                    continue
            claims_idx.extend(len(item['unreflected_answer_claim']) * [idx + 1] + len(item['reflected_answer_claim']) * [- idx - 1])
            all_claims = item['unreflected_answer_claim'] + item['reflected_answer_claim']
            claims.extend(all_claims)
            input_question = item['prompt']
            inputs.extend([input_question] * len(all_claims))
            for claim in all_claims:
                passages = retriever.get_passages(entity, claim, args.top_k)
                # retriever.save_cache()
                context = ""
                for i, passage in enumerate(passages):
                    if 'enwiki' in args.db:
                        psg = passage['text']
                    else:
                        psg = passage
                    if i > 0:
                        context += "\n---\n" + psg.strip(' \n')
                    else:
                        context += psg.strip(' \n')
                prompts.append(USER_PROMPT.format(input=input_question, claim=claim, context=context))
        with open(args.prompt_file, 'w') as f:
            json.dump({'claims_idx': claims_idx, 'prompts': prompts, 'inputs': inputs, 'claims': claims}, f)
    else:
        with open(args.prompt_file, 'r') as f:
            prompt_obj = json.load(f)
        claims_idx = prompt_obj['claims_idx']
        prompts = prompt_obj['prompts']
        inputs = prompt_obj['inputs']
        claims = prompt_obj['claims']

    async def api_call():
        messages = [[{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': p}] for p in prompts]
        cache_path_fc = os.path.join('openai_cache', f"{args.cache_name}_fc.diskcache")
        responses = await create_answers_async(MODEL_DICT[args.llm], messages, cache_path_fc, batch_size=args.llm_batch_size)
        # responses = ask_open_ai(args, prompts, batch_result=args.batch_result_fc, sub_cache_name='fc', real_time=args.fc_real_time)
        summary_prompts = [SUMMARY_PROMPT.format(claim=c, input=i, reply=response) for c, i, response in zip(claims, inputs, responses)]
        summary_messages = [[{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': p}] for p in summary_prompts]
        cache_path_sum = os.path.join('openai_cache', f"{args.cache_name}_sum.diskcache")
        summaries = await create_answers_async(MODEL_DICT[args.llm], summary_messages, cache_path_sum, batch_size=args.llm_batch_size)
        return responses, summaries
    # summaries = ask_open_ai(args, summary_prompts, batch_result=args.batch_result_sum, sub_cache_name='sum', real_time=args.sum_real_time)

    responses, summaries = asyncio.run(api_call())

    assert(len(summaries) == len(claims_idx))
    for c_id, summary, response in zip(claims_idx, summaries, responses):
        if c_id > 0:
            data_objs[c_id - 1]['unreflected_claim_fc'].append(response)
            data_objs[c_id - 1]['unreflected_claim_sum'].append(summary)
        else:
            data_objs[- (c_id + 1)]['reflected_claim_fc'].append(response)
            data_objs[- (c_id + 1)]['reflected_claim_sum'].append(summary)

    with open(args.output_file, 'w') as f:
        for obj in data_objs:
            f.write(json.dumps(obj))
            f.write('\n')


if __name__ == '__main__':
    main()
