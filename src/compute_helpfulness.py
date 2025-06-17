import time
import diskcache as dc
import asyncio
import openai
from openai import OpenAI, AsyncOpenAI
import tiktoken
import json
import random
import pandas as pd
import os
import argparse
from tqdm import tqdm
from datasets import load_dataset

SYSTEM_PROMPT_JUDGE = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two "
    "AI assistants to the user question displayed below. You should choose the assistant that "
    "follows the user's instructions and answers the user's question better. Your evaluation "
    "should focus on factors such as the helpfulness, relevance, depth, "
    "and level of detail of their responses. Do not take correctness into consideration."
    " Begin your evaluation by comparing the two "
    "responses and provide a short explanation. Avoid any position biases and ensure that the "
    "order in which the responses were presented does not influence your decision. Do not allow "
    "the length of the responses to influence your evaluation. Do not favor certain names of "
    "the assistants. Be as objective as possible. After providing your explanation, output your "
    'final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" '
    '"if assistant B is better, and "[[C]]" for a tie.'
)

USER_PROMPT = """
### User's Question:
{question}

<|The Start of Assistant A's Response to the User|>
{answer_a}
<|The End of Assistant A's Response to the User|>
<|The Start of Assistant B's Response to the User|>
{answer_b}
<|The End of Assistant B's Response to the User|>"""

MODEL_DICT = {
    'gpt-4o-1': 'gpt-4o',
}
INPUT_COST_DICT = {
    'gpt-4o-mini': 0.15,
    'gpt-4o-1': 2.5,
}
OUTPUT_COST_DICT = {
    'gpt-4o-mini': 0.6,
    'gpt-4o-1': 10,
}


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
        system_prompt: str = SYSTEM_PROMPT_JUDGE,
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


def ask_open_ai(args, prompts, batch_result, sub_cache_name, system_prompt=SYSTEM_PROMPT_JUDGE, real_time=False):
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
            for p in tqdm(prompts, total=len(prompts), desc='OpenAI Chat'):
                responses.append(openai_chat.ask(p))
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
            # if args.cache_name + f'_{sub_cache_name}_' + str(idx) in response_dict.keys():
            responses.append(response_dict[args.cache_name + f'_{sub_cache_name}_' + str(idx)])
            # else:
            #     responses.append('')
    output_costs = 0
    for r in responses:
        output_costs += get_output_price(r, args.llm)
    if real_time:
        print(f"Output cost with Batch API", output_costs)
    else:
        print(f"Output cost with Batch API", output_costs / 2)
    return responses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assistant_a", type=str, default='')
    parser.add_argument("--assistant_b", type=str, default='')
    parser.add_argument("--data", type=str, default='bio')
    parser.add_argument("--llm", type=str, default="gpt-4o-1", choices=['gpt-4o-mini', 'gpt-4o-1'])
    parser.add_argument("--cache_name", type=str, default='', help='Cache name for OpenAI batch API')
    parser.add_argument("--batch_result", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--real_time", default=False, action='store_true')
    parser.add_argument("--sample", type=int, default=-1)
    args = parser.parse_args()

    assistant_a = []
    questions = []
    with open(args.assistant_a) as f:
        for line in f.readlines():
            obj = json.loads(line)
            response = obj['response'] if '<reflection>' not in obj['response'] else obj['response'].split('<reflection>')[0].strip('\n ')
            response = response.replace('<|endoftext|>', '')
            assistant_a.append(response)
            if args.data == 'bio':
                questions.append(f"Tell me a bio of {obj['entity']}.")
            else:
                questions.append(f"In a paragraph, could you tell me what you know about {obj['entity']}")

    assistant_b = []
    with open(args.assistant_b) as f:
        for line in f.readlines():
            obj = json.loads(line)
            response = obj['response'] if '<reflection>' not in obj['response'] else obj['response'].split('<reflection>')[0].strip('\n ')
            response = response.replace('<|endoftext|>', '')
            assistant_b.append(response)

    print(len(assistant_a), len(assistant_b), len(questions))

    prompts_to_run = []
    for answer_a, answer_b, prompt in zip(assistant_a, assistant_b, questions):
        prompts_to_run.append(USER_PROMPT.format(question=prompt, answer_a=answer_a, answer_b=answer_b))
        prompts_to_run.append(USER_PROMPT.format(question=prompt, answer_a=answer_b, answer_b=answer_a))

    # print(prompts_to_run[0])

    responses = ask_open_ai(args, prompts_to_run, batch_result=args.batch_result, system_prompt=SYSTEM_PROMPT_JUDGE, sub_cache_name='h', real_time=args.real_time)

    forward_responses = [response for i, response in enumerate(responses) if i % 2 == 0]
    backward_responses = [response for i, response in enumerate(responses) if i % 2 == 1]
    assert len(forward_responses) == len(backward_responses)

    final_judge = []
    for response, reverse_response, question, answer_a, answer_b in zip(forward_responses, backward_responses, questions, assistant_a, assistant_b):
        if '[[A]]' in response and '[[B]]' in reverse_response:
            final_judge.append('A')
        elif '[[B]]' in response and '[[A]]' in reverse_response:
            final_judge.append('B')
        elif '[[A]]' in response and '[[C]]' in reverse_response:
            final_judge.append('A')
        elif '[[B]]' in response and '[[C]]' in reverse_response:
            final_judge.append('B')
        elif '[[C]]' in response and '[[B]]' in reverse_response:
            final_judge.append('A')
        elif '[[C]]' in response and '[[A]]' in reverse_response:
            final_judge.append('B')
        else:
            final_judge.append('C')

    # print("Tie portion", final_judge.count('C') / len(final_judge))
    # print(f"{args.assistant_a} portion", final_judge.count('A') / len(final_judge))
    # print(f"{args.assistant_b} portion", final_judge.count('B') / len(final_judge))
    print("Score", (final_judge.count('B') + final_judge.count('C') / 2) / len(final_judge))
    pd.DataFrame({'prompt': questions, 'final_judge': final_judge, 'response_a': assistant_a, 'response_b': assistant_b, 'forward': forward_responses, 'backward': backward_responses}).to_excel(args.batch_result.replace('.json', '.xlsx') if args.batch_result else args.output_file, index=False)


if __name__ == '__main__':
    main()
