import os
import pickle
import gc
import numpy as np
import torch
import pandas as pd
import yaml
import json  # Importing json module
from argparse import ArgumentParser
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import pprint  # For pretty-printing the config

def filter_lora(lora_path):
    original_ckpt_folder = lora_path
    new_ckpt_folder = lora_path + '-filtered'
    # print(original_ckpt_folder)
    # print(new_ckpt_folder)

    if not os.path.exists(new_ckpt_folder):
        shutil.copytree(original_ckpt_folder, new_ckpt_folder)
    
    # adapter_path1 = new_ckpt_folder +'/adapter_model/adapter_model.safetensors'
    adapter_path2 =  new_ckpt_folder + '/adapter_model.safetensors' # This is more important
    # print(adapter_path1)
    # print(adapter_path2)

    tensors =  safetensors.torch.load_file(adapter_path2)

    nonlora_keys = []
    for k in list(tensors.keys()):
        if "lora" not in k:
            nonlora_keys.append(k)
    # print(nonlora_keys) # just take a look what they are

    for k in nonlora_keys:
        del tensors[k]

    # safetensors.torch.save_file(tensors, adapter_path1)
    safetensors.torch.save_file(tensors, adapter_path2)
    return new_ckpt_folder

class Generator:
    def __init__(self, model_type, llm, tokenizer):
        self.model_type = model_type
        self.llm = llm
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate_responses(self, examples, sampling_params, lora_request=None, format_prompt=False, system_prompt=""):
        """Generates responses based on provided sampling parameters."""
        if format_prompt:
            prompts = [self._format_prompt(system_prompt, p, no_system=False) for p in examples]
        else:
            prompts = examples

        response = self.llm.generate(prompts, sampling_params, lora_request=lora_request)
        return response

    def _format_prompt(self, system, input_text, no_system=False):
        if no_system:
            chat = [{"role": "user", "content": system + '\n\n' + input_text}]
        else:
            chat = [
                {"role": "system", "content": system},
                {"role": "user", "content": input_text},
            ]
        if self.tokenizer.chat_template is None:
            raise ValueError("The tokenizer does not have a chat template.")
        formatted_input = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return formatted_input

def load_datasets(file_paths, question_columns):
    if not isinstance(file_paths, list):
        file_paths = [file_paths]
    if not isinstance(question_columns, list):
        question_columns = [question_columns]

    if len(question_columns) == 1:
        question_columns = question_columns * len(file_paths)
    elif len(question_columns) != len(file_paths):
        raise ValueError("The number of 'question_column' entries must match the number of datasets provided, or be a single value.")

    datasets = []
    question_lists = []
    for file_path, question_column in zip(file_paths, question_columns):
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        elif file_extension == '.json':
            df = pd.read_json(file_path)
        elif file_extension == '.jsonl':
            df = pd.read_json(file_path, lines=True)
        elif file_extension in ['.pkl', '.pickle']:
            with open(file_path, 'rb') as f:
                df = pickle.load(f)
            if isinstance(df, list):
                df = pd.DataFrame(df)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

        # Verify that the question column exists
        if question_column not in df.columns:
            raise ValueError(f"Column '{question_column}' not found in dataset '{file_path}'.")

        datasets.append(df)
        # Build question list from the specified column
        questions = []
        for item in df[question_column]:
            if isinstance(item, list):
                questions.extend(item)
            else:
                questions.append(item)
        question_lists.extend(questions)

    combined_dataset = pd.concat(datasets, ignore_index=True)
    return combined_dataset, question_lists

def merge_configs(config, args, parser):
    # Command-line arguments take precedence over config file
    for key, value in vars(args).items():
        default = parser.get_default(key)
        if value != default:
            config[key] = value
    return config

def main(config):
    # Ensure that the necessary environment variables are set
    hf_token = os.environ.get('HF_TOKEN')
    if hf_token is None:
        raise ValueError("Please set the 'HF_TOKEN' environment variable.")

    # Load tokenizer
    tokenizer_dir = config.get('tokenizer_path', config['model_type'])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, padding_side='left', local_files_only=False)
    print(f'Loaded tokenizer from {tokenizer_dir}')

    stop_tokens = list(set([tokenizer.eos_token]+config.get('stop_tokens', [])))
    # print(stop_tokens)
    print(f'Stop tokens: {stop_tokens}')
    # Load datasets and question lists
    print('Loading datasets')
    dataset_paths = config['dataset_path']
    if not isinstance(dataset_paths, list):
        dataset_paths = [dataset_paths]

    question_columns = config.get('question_column', 'instruction')
    if not isinstance(question_columns, list):
        question_columns = [question_columns]

    # Validate question columns
    if len(question_columns) == 1:
        question_columns = question_columns * len(dataset_paths)
    elif len(question_columns) != len(dataset_paths):
        raise ValueError("The number of 'question_column' entries must match the number of datasets provided, or be a single value.")

    dataset, question_list = load_datasets(dataset_paths, question_columns)

    # Use subset if specified
    if config.get('subset', False):
        subset_num = config.get('subset_num', 10)
        question_list = question_list[:subset_num]
        print(f'Running on subset of {subset_num} questions')
    else:
        print(f'Total number of questions: {len(question_list)}')

    # Split the question list into batches
    by_batch = config.get('by_batch', 1)
    operate_batch_num = config.get('operate_batch_num', 1)
    question_list_batches = np.array_split(question_list, by_batch)
    # Select the specified batch
    question_list = question_list_batches[operate_batch_num - 1]
    print(f'Operating on batch {operate_batch_num} of {by_batch}, batch size: {len(question_list)}')

    # Load model
    print('Occupying GPU memory')
    device = torch.device("cuda:0")
    # make sure any old allocations are cleared
    torch.cuda.empty_cache()

    # figure out how many bytes 70% is
    props      = torch.cuda.get_device_properties(device)
    total_mem  = props.total_memory
    target_mem = int(total_mem * 0.7)

    # allocate in 1 GB chunks until we've claimed ~70%
    allocated = 0
    tensors  = []
    chunk_mb  = 1024**5  # 1 GB
    while allocated < target_mem:
        this_chunk = min(target_mem - allocated, chunk_mb)
        tensors.append(torch.empty(this_chunk, dtype=torch.uint8, device=device))
        allocated += this_chunk

    # drop them immediately to free for your real model
    del tensors
    # torch.cuda.empty_cache()


    print('Loading model')
    llm = LLM(
        model=config['model_type'],
        dtype='auto',
        seed=config.get('seed', 42),
        trust_remote_code=True,
        enable_lora=bool(config.get('lora_checkpoint')),
        max_lora_rank=64,
        tensor_parallel_size=config.get('tensor_parallel_size', 1),
        gpu_memory_utilization=config.get('gpu_memory_utilization', 0.8)  # Adjust if necessary
    )
    generator = Generator(config['model_type'], llm, tokenizer)
    print('Finished loading model')
    torch.cuda.empty_cache()

    # Load LoRA checkpoint if provided
    lora_checkpoint = config.get('lora_checkpoint')
    if lora_checkpoint:
        print('Using LoRA checkpoint')
        new_lora_checkpoint = filter_lora(lora_checkpoint)
        lora_request = LoRARequest('lora', 1, new_lora_checkpoint)
    else:
        print('Not using LoRA')
        lora_request = None

    # Set up sampling parameters

    sampling_params = SamplingParams(
        n=config.get('num_sampling', 1),
        temperature=config.get('temperature', 0.7),
        top_p=config.get('top_p', 1.0),
        top_k=config.get('top_k', 50),
        seed=config.get('seed', 42),
        logprobs=config.get('logprobs'),
        max_tokens=config.get('max_tokens', 100),
        stop=stop_tokens,
        include_stop_str_in_output=True,
        skip_special_tokens=False,
        repetition_penalty = config.get('repetition_penalty', 1.0),
        frequency_penalty = config.get('frequency_penalty', 0.0),
    )


    # Generate responses
    print(f'Generating responses with sampling parameters: {sampling_params}')
    response = generator.generate_responses(
        examples=question_list,
        sampling_params=sampling_params,
        lora_request=lora_request,
        format_prompt=config.get('format_prompt', False),
        system_prompt=config.get('system_prompt', "")
    )

    # Clean up
    del generator
    del llm
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare results for JSONL saving
    results = []
    for request_output in response:
        entry = {}
        entry['prompt'] = request_output.prompt
        entry['responses'] = [generation.text for generation in request_output.outputs]
        results.append(entry)

    # Determine the output file name or use the user-provided one
    if config.get('output_file'):
        filename = config['output_file']
        print(f'Saving results to {filename}')
    else:
        model_used = os.path.basename(config['model_type'])
        data_used = '_'.join([os.path.splitext(os.path.basename(path))[0] for path in dataset_paths])

        # Prepare filename with batch info
        filename = f'../checkpoint_responses/response_{data_used}_{config.get("num_sampling", 1)}_{model_used}'
        if config.get('subset', False):
            filename = f'../checkpoint_responses/subset_{data_used}_{config.get("num_sampling", 1)}_{model_used}'
        if by_batch > 1:
            filename += f'_batch_{operate_batch_num}_of_{by_batch}'
        filename += '.jsonl'  # Change extension to .jsonl


    # os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as file:
        for entry in results:
            json_line = json.dumps(entry, ensure_ascii=False)
            file.write(json_line + '\n')
    print(f'Saved results to {filename}')

if __name__ == "__main__":
    os.environ['HF_TOKEN']="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"
    parser = ArgumentParser(description='Model inference script.')
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file.')


    # Command-line arguments (they take precedence over config file)
    parser.add_argument("--num_sampling", type=int, help="Number of times to sample.")
    parser.add_argument("--lora_checkpoint", help="Path to LoRA checkpoint.")
    parser.add_argument("--dataset_path", type=str, nargs='+', help="Path(s) to the dataset file(s).")
    parser.add_argument("--model_type", type=str, help="Type or path of the model to use.")
    parser.add_argument("--tokenizer_path", type=str, help="Path to the tokenizer. Defaults to model_type if not provided.")
    parser.add_argument("--subset", action="store_true", help="Use a subset of the data.")
    parser.add_argument("--subset_num", type=int, help="Number of samples in the subset.")
    parser.add_argument("--tensor_parallel_size", type=int, help="Tensor parallel size for the model.")
    parser.add_argument("--by_batch", type=int, help="Number of batches to split the data into.")
    parser.add_argument("--operate_batch_num", type=int, help="Batch number to operate on.")
    parser.add_argument("--question_column", type=str, nargs='+', help="Column name(s) to use for building the question list.")
    parser.add_argument("--output_file", type=str, help="Optional path to save the output file.")

    # Sampling parameters
    parser.add_argument("--temperature", type=float, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, help="Top-p (nucleus) sampling.")
    parser.add_argument("--top_k", type=int, help="Top-k sampling.")
    parser.add_argument("--max_tokens", type=int, help="Maximum number of tokens to generate.")
    parser.add_argument("--seed", type=int, help="Random seed for sampling.")
    parser.add_argument("--logprobs", type=int, help="Number of logprobs to return.")
    parser.add_argument("--format_prompt", action="store_true", help="Whether to format the prompt.")
    parser.add_argument("--system_prompt", type=str, help="System prompt to prepend.")
    parser.add_argument("--stop_tokens", nargs='*', help="List of stop tokens.")
    args = parser.parse_args()

    # Load configuration from YAML file if provided
    # Load configuration from YAML file if provided
    print(f'using model: {args.model_type}')
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
        
    print(config)

    # Merge configurations, command-line arguments take precedence
    config = merge_configs(config, args, parser)

    # Print the configuration after merging
    print("\nConfiguration after merging:")
    pprint.pprint(config)
    print()

    # Check for required parameters
    required_params = ['dataset_path', 'model_type']
    for param in required_params:
        if param not in config or config[param] is None:
            parser.error(f"The following argument is required: --{param.replace('_', '-')}")

    main(config)
