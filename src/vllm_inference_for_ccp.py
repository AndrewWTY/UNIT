import argparse
import json

import pandas as pd
import random
import pickle
from typing import List, Dict

from transformers import AutoTokenizer, set_seed, logging
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from huggingface_hub import login


login(token="replace_with_your_huggingface_token")  # Set your Hugging Face token here


MMLU_SYSTEM = """You are a helpful and honest AI assistant who tells what you know and what you don't know."""
SYSTEM = """You are a helpful AI assistant."""
SYSTEM_LIN = """You are a helpful and honest AI assistant. 
Whenever the user gives you an **information-seeking** instruction, you always express your uncertainty about the facts that you are not sure of. For knowledge that you are certain about, you don't need to add uncertainty expressions.
"""

SYSTEM_REF = """You are a helpful and honest AI assistant. 
Whenever the user gives you an **information-seeking** instruction, you always include a <reflection> section after your answer to highlight any of your uncertainties. For knowledge that you are certain about, you never include them in the <reflection>.
"""


class VllmProbsCalculator:
    """
    For Whitebox model (lm_polygraph.WhiteboxModel), at (input text, target text) batch calculates:
    * probabilities distribution of tokens in the generation target.
    """

    def __init__(self, n_alternatives: int = 10):
        self.n_alternatives = n_alternatives

    def __call__(
            self,
            prompts: List[str],
            model,
            sampling_parameters: SamplingParams,
            lora_path=None,
            **kwargs,
    ) -> Dict[str, List]:
        """
        Calculates the statistics of probabilities at each token position in the output text.

        Parameters:
            input_texts (List[str]): Input texts batch.
            output_texts (List[str]): Output texts batch.
            model (Model): Model used for generation.
        """
        sampling_parameters.logprobs = self.n_alternatives
        sampling_parameters.temperature = 0

        if lora_path:
            outputs = model.generate(prompts, sampling_parameters, lora_request=LoRARequest("lora", 1, lora_path))
        else:
            outputs = model.generate(prompts, sampling_parameters)

        log_probs = []
        token_ids = []
        raw_texts = []
        alternatives = []
        for output in outputs:
            output_sample = output.outputs[0]
            token_ids.append(list(output_sample.token_ids))
            raw_texts.append(output_sample.text)
            log_probs.append(
                [l[output_sample.token_ids[idx]].logprob for idx, l in enumerate(output_sample.logprobs)])
            alternative_token = []
            for idx, l in enumerate(output_sample.logprobs):
                alternative_token.append([])
                for k, v in l.items():
                    lp = v.logprob
                    alternative_token[-1].append((k, lp))
                alternative_token[-1] = sorted(
                    alternative_token[-1],
                    key=lambda x: x[0] == output_sample.token_ids[idx],
                    reverse=True
                )
            alternatives.append(alternative_token)

        result_dict = {
            "greedy_tokens": token_ids,
            "greedy_tokens_alternatives": alternatives,
            "greedy_texts": raw_texts,
            "greedy_log_likelihoods": log_probs,
        }

        return result_dict


def format_prompt(tokenizer, system, input, no_system=False):
    if no_system:
        chat = [
            {"role": "user", "content": system + '\n\n' + input},
        ]
    else:
        chat = [
            {"role": "system", "content": system},
            {"role": "user", "content": input},
        ]
    formatted_input = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return formatted_input


def capitalize_first_letter(input_string):
    if not input_string:
        return input_string
    if input_string[0].islower():
        return input_string[0].upper() + input_string[1:]
    return input_string


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--model_cache_dir", type=str, default="cache")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--architecture", type=str, default="llama-3")
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument("--prompt_file", type=str, default="example_prompts.json")
    parser.add_argument("--instruction_field", type=str, default="instruction")
    parser.add_argument("--output_file", type=str, default="output_greedy_probs.json")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_tokenizer", action="store_true", default=False)
    return parser.parse_args()


def main(args):
    if args.load_tokenizer:
        tokenizer_dir = args.model_path if args.lora_path is None else args.lora_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, cache_dir=args.model_cache_dir,
                                                  padding_side='left', local_files_only=False)
        tokenizer_name = args.model_path
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, cache_dir=args.model_cache_dir,
                                                  padding_side='left', local_files_only=False)
        tokenizer_name = args.tokenizer_path

    if args.system_prompt is None:
        sys = SYSTEM
    elif "ref" in args.system_prompt:
        sys = SYSTEM_REF
    elif "mmlu" in args.system_prompt:
        sys = MMLU_SYSTEM
    else:
        sys = SYSTEM_LIN

    if args.prompt_file.endswith('csv'):
        df = pd.read_csv(args.prompt_file)
    elif args.prompt_file.endswith('xlsx'):
        df = pd.read_excel(args.prompt_file)
    elif args.prompt_file.endswith('json'):
        df = pd.read_json(args.prompt_file)
    else:
        if args.prompt_file.endswith('jsonl'):
            lines = True
        else:
            lines = False
        df = pd.read_json(args.prompt_file, lines=lines)
    instructions = df[args.instruction_field].to_list()
    instructions = [capitalize_first_letter(i) for i in instructions]

    if 'gemma' in args.architecture:
        prompts = [format_prompt(tokenizer, sys, p, no_system=True) for p in instructions]
        eos_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<end_of_turn>')]
        stops = None
    elif 'llama-3' in args.architecture:
        prompts = [format_prompt(tokenizer, sys, p) for p in instructions]
        eos_token_ids = [tokenizer.eos_token_id]
        if 'instruct' in args.architecture.lower():
            eos_token_ids.append(tokenizer.convert_tokens_to_ids('<|eot_id|>'))
            stops = None
        else:
            stops = ['<|im_end|>']
    elif 'phi3' in args.architecture:
        prompts = [format_prompt(tokenizer, sys, p) for p in instructions]
        eos_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|end|>')]
        stops = None
    else:
        prompts = [format_prompt(tokenizer, sys, p) for p in instructions]
        eos_token_ids = [tokenizer.eos_token_id]
        stops = ['<|im_end|>']

    if args.sample:
        prompts = prompts[:args.sample]

    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        seed=args.seed,
        logprobs=10,
        stop_token_ids=eos_token_ids,
        stop=stops,
        max_tokens=512,
        include_stop_str_in_output=False,
    )
    llm = LLM(
        model=args.model_path,
        enable_lora=True if args.lora_path is not None else False,
        download_dir=args.model_cache_dir,
        tokenizer=tokenizer_name,
        dtype='auto',
        seed=args.seed,
        trust_remote_code=True,
        max_lora_rank=64,
    )

    probs_cal = VllmProbsCalculator(n_alternatives=10)
    greedy_probs = probs_cal(prompts=prompts, model=llm, sampling_parameters=sampling_params)
    greedy_probs['input_text'] = prompts
    with open(args.output_file, 'w') as f:
        json.dump(greedy_probs, f)


if __name__ == "__main__":
    args = get_args()

    if args.seed >= 0:
        set_seed(args.seed)
        random.seed(args.seed)

    logging.set_verbosity_info()

    main(args)



