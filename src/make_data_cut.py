import json
import os
import diskcache as dc
import time
import asyncio
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from litellm import completion, acompletion


CERTAIN_CONCAT_PROMPT = """[Instruction]: "{instruction}"

[Fact List]: \"\"\"
{fact_list}\"\"\"

Please concatenate the facts from the [Fact List] to form a helpful [Response] to the [Instruction].

Important Requirements:
1. Make sure your [Response] sounds helpful, fluent, and natural. Use logical conjunctions frequently.
2. Do not add new fact or information except from those in [Fact List].
3. Make sure to involve all information in [Fact List].

[Response]:
"""

SYSTEM_MSG = "You are a helpful assistant.you should answer user\u2019s query directly, providing a helpful and accurate response to the query."

MODEL_DICT = {
    'o1': 'o1',
    'o3-mini': 'o3-mini',
    'gpt-4o-mini': 'gpt-4o-mini',
    'gpt-4o': 'gpt-4o',
    'deepseek-chat': 'together_ai/deepseek-ai/DeepSeek-V3',
    'deepseek-reasoner': "together_ai/deepseek-ai/DeepSeek-R1",
    'qwq-32b': 'together_ai/Qwen/QwQ-32B',
    'sonnet-3.7-high': "anthropic/claude-3-7-sonnet-20250219",
    'gemini-2.5-pro': "gemini/gemini-2.5-pro-preview-03-25",
    'llama_405': "together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    'llama_70': "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    'llama4_maverick': "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    'gemma3': "together_ai/google/gemma-3-12b-it",
}
INPUT_COST_DICT = {
    'o1': 15,
    'o3-mini': 1.1,
    'gpt-4o-mini': 0.15,
    'gpt-4o': 2.5,
    # 'deepseek-chat': 0.27,
    # 'deepseek-reasoner': 0.55,
    'deepseek-chat': 1.25,
    'deepseek-reasoner': 3,
    'qwq-32b': 1.2,
    'sonnet-3.7-high': 3,
    'gemini-2.5-pro': 1.25,
    'llama_405': 3.5,
    'llama_70': 0.88,
    'llama4_maverick': 0.27,
    'gemma3': 0.3,
}
OUTPUT_COST_DICT = {
    'o1': 60,
    'o3-mini': 4.4,
    'gpt-4o-mini': 0.6,
    'gpt-4o': 10,
    # 'deepseek-chat': 1.10,
    # 'deepseek-reasoner': 2.19,
    'deepseek-chat': 1.25,
    'deepseek-reasoner': 7,
    'qwq-32b': 1.2,
    'sonnet-3.7-high': 15,
    'gemini-2.5-pro': 2.5,
    'llama_405': 3.5,
    'llama_70': 0.88,
    'llama4_maverick': 0.85,
    'gemma3': 0.3,
}
GENE_ARGS_DICT = {
    'gpt-4o': {'temperature': 0, 'max_tokens': 4096},
    'gpt-4o-mini': {'temperature': 0, 'max_tokens': 4096},
    'deepseek-reasoner': {'temperature': 0.6, 'max_tokens': 8192},
    'qwq-32b': {'temperature': 0.6, 'top_p': 0.95, 'max_tokens': 8192},
    'o3-mini': {'reasoning_effort': 'medium'},
    'sonnet-3.7-high': {'reasoning_effort': 'high'},
    'gemini-2.5-pro': {},
    'deepseek-chat': {'temperature': 0, 'max_tokens': 4096},
    'llama_405': {'temperature': 0, 'max_tokens': 4096},
    'llama_70': {'temperature': 0, 'max_tokens': 4096},
    'llama4_maverick': {'temperature': 0, 'max_tokens': 4096},
    'gemma3': {'temperature': 0, 'max_tokens': 4096},
}


def get_input_price(model, input_len=None):
    input_cost = input_len / 1000000 * INPUT_COST_DICT[model]
    return input_cost


def get_output_price(model, output_len=None):
    output_cost = output_len / 1000000 * OUTPUT_COST_DICT[model]
    return output_cost


def print_claims(claims, number=True):
    claim_list = ""
    if number:
        for i, c in enumerate(claims):
            claim_list += f'{i + 1}. {c}\n'
    else:
        for _, c in enumerate(claims):
            claim_list += f'- {c}\n'
    return claim_list


class LiteLLMChat:
    def __init__(
            self,
            model_name: str = None,
            cache_path: str = "litellm_cache",
            cache_name: str = "cache",
            generation_args: dict = None,
    ):
        self.model_name = model_name
        self.cache_path = os.path.join(cache_path, f"{cache_name}.diskcache")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        self.generation_args = generation_args

    def ask(self, message: str):
        cache_settings = dc.DEFAULT_SETTINGS.copy()
        cache_settings["eviction_policy"] = "none"
        cache_settings["size_limit"] = int(1e12)
        cache_settings["cull_limit"] = 0
        with dc.Cache(self.cache_path, **cache_settings) as litellm_responses:
            if (self.model_name, message) in litellm_responses:
                reply_content = litellm_responses[(self.model_name, message)]
                print("Loaded from cache")
                input_price, output_price, input_token_num, output_token_num = 0, 0, 0, 0
            else:
                messages = [{"role": "user", "content": message}]
                chat = self._send_request(messages)
                reply_content = {
                    'response': chat.choices[0].message.content,
                    'response_reasoning': chat.choices[0].message.reasoning_content,
                }
                litellm_responses[(self.model_name, message)] = reply_content
            input_token_num = chat.usage.prompt_tokens
            input_price = get_input_price(self.model_name, input_token_num)
            output_token_num = chat.usage.completion_tokens
            output_price = get_output_price(self.model_name, output_token_num)

        return reply_content, input_price, input_token_num, output_price, output_token_num

    def _send_request(self, messages):
        sleep_time_values = (5, 10, 30, 60, 120)
        arg_dict = {
            'model': self.model_name,
            'messages': messages,
            **self.generation_args,
        }
        for i in range(len(sleep_time_values)):
            try:
                return completion(**arg_dict)
            except Exception as e:
                sleep_time = sleep_time_values[i]
                print(
                    f"Request to LiteLLM failed with exception: {e}. Retry #{i}/5 after {sleep_time} seconds."
                )
                time.sleep(sleep_time)

        return completion(**arg_dict)


async def achat(model, messages, generation_args):
    output = await acompletion(model=MODEL_DICT[model], messages=messages, **generation_args)
    input_token_num = output.usage.prompt_tokens
    output_token_num = output.usage.completion_tokens
    try:
        reasoning_content = output.choices[0].message.reasoning_content
    except Exception as e:
        reasoning_content = None
    return output.choices[0].message.content, reasoning_content, input_token_num, output_token_num


def batchify(lst, batch_size):
    """Split the list `lst` into sublists of size `batch_size`."""
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


async def create_answers_async(model, messages, cache_path, generation_args, batch_size=5):
    # async answering
    batched_msgs = batchify(messages, batch_size)
    total_input_tok_num = 0
    total_output_tok_num = 0
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
                    api_responses = await asyncio.gather(*[achat(model, m, generation_args) for m in cache_miss_msgs])
                    answers, reasoning_contents, input_tok_nums, output_tok_nums = zip(*api_responses)
                    total_input_tok_num += sum(input_tok_nums)
                    total_output_tok_num += sum(output_tok_nums)
                    for msg, res, reasoning in zip(cache_miss_msgs, answers, reasoning_contents):
                        litellm_responses[(model, msg)] = {'response': res, 'response_reasoning': reasoning}
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

    input_price = get_input_price(model, total_input_tok_num)
    output_price = get_output_price(model, total_output_tok_num)
    return all_answers, error_batches, input_price + output_price


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--cache_name", type=str, default='openai')
    parser.add_argument("--llm", type=str, default="gpt-4o")
    parser.add_argument("--threshold_key", type=str, default="q75")
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    raw_data_to_surgery = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            raw_data_to_surgery.append(json.loads(line))

    def is_info(d):
        return d.get("primary_tag", "").lower() == "information seeking" and d.get(
            "other_tags"
        ) is None

    data_to_surgery = []
    for item in raw_data_to_surgery:
        item['info-seeking'] = is_info(item)
        certain_claims = []
        uncertain_claims = []
        if item['claim_uncertainty']:
            for claim in item['claim_uncertainty']:
                if claim[args.threshold_key] == 1:
                    uncertain_claims.append(claim['claim'])
                else:
                    certain_claims.append(claim['claim'])
        item['certain_claims'] = certain_claims
        item['uncertain_claims'] = uncertain_claims
        data_to_surgery.append(item)

    prompts = []
    surgery_ids = []
    no_certain_ids = []
    for idx, item in enumerate(data_to_surgery):
        if item['info-seeking'] and len(item['certain_claims']) > 0 and len(item['uncertain_claims']) > 0:
            prompts.append([
                {'role': 'system', 'content': "You are a helpful AI assistant."},
                {'role': 'user', 'content': CERTAIN_CONCAT_PROMPT.format(instruction=item['prompt'], fact_list=print_claims(item['certain_claims']))},
            ])
            surgery_ids.append(idx)
        elif item['info-seeking'] and len(item['certain_claims']) == 0 and len(item['uncertain_claims']) > 0:
            no_certain_ids.append(idx)

    total_cost = 0
    responses, err_batches, cost = await create_answers_async(
        args.llm,
        prompts,
        cache_path=os.path.join('litellm_cache', f"{args.cache_name}.diskcache"),
        generation_args=GENE_ARGS_DICT[args.llm],
        batch_size=args.batch_size,
    )
    total_cost += cost
    print("Error Batches", err_batches)
    print(f"Total cost {total_cost}")

    surgery_responses = [item['response'] for item in data_to_surgery]
    for idx, s_id in enumerate(surgery_ids):
        surgery_responses[s_id] = responses[idx]

    for idx, n_id in enumerate(no_certain_ids):
        surgery_responses[n_id] = "Sorry but I am unconfident to precisely answer your question."

    formatted_messages = [
        {"messages": [
            {'role': 'system', 'content': SYSTEM_MSG},
            {'role': 'user', 'content': item['prompt']},
            {'role': 'assistant', 'content': res},
        ]} for item, res in zip(data_to_surgery, surgery_responses)
    ]

    with open(args.output_file, 'w') as f:
        for msg in formatted_messages:
            f.write(json.dumps(msg))
            f.write('\n')


if __name__ == '__main__':
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
