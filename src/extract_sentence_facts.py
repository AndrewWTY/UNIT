import json
import re
import argparse
import pandas as pd
import pickle
import asyncio
from openai import AsyncOpenAI, OpenAI
from instruction_classification import create_answers_async, get_input_price, MODEL_DICT, get_output_price
import stanza
import platform
import os
# stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize')

EXTRACTION_PROMPT = """Break down the following sentence into atomic facts.
___
{sentence}
___

Respond with the following format:

- <atomic fact 1>
- <atomic fact 2>
...

However, if there is no factual claim, respond <EMPTY>."""


MATCHING_PROMPT = """Given the fact, identify the corresponding words in the original sentence that help derive this fact. Please list all words that are related to the fact, in the order they appear in the original sentence, each word separated by comma.
Fact: {claim}
Sentence: {sent}
Words from sentence that helps to derive the fact, separated by comma: """


def ask_open_ai(args, prompts, sub_cache_name, batch_result, system_msg="You are a helpful AI assistant."):
    input_prices = []
    parsed_prompts = []
    # import pdb; pdb.set_trace()
    for p in prompts:
        message = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": p},
        ]
        parsed_prompts.append(message)
        input_prices.append(get_input_price(system_msg + p, args.llm))

    print(f"Input cost for {sub_cache_name} step", sum(input_prices) / ((not args.real_time) + 1))
    if args.real_time:
        async_client = AsyncOpenAI()

        responses = asyncio.run(create_answers_async(async_client, model=MODEL_DICT[args.llm], messages=parsed_prompts,
                                                     batch_size=args.batch,
                                                     seed=args.seed, temperature=0,
                                                     cache_name=args.cache_name + f'_{sub_cache_name}'))
    else:
        responses = []
        # Use batch API
        if batch_result is None:
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
                            "seed": args.seed,
                            "temperature": 0,
                        }
                    })

                # Write batch prompts to file
                batch_prompt_filename = '../evaluate_database/'+ args.cache_name + f'_{sub_cache_name}_batch_prompt_{chunk_start}_{chunk_end}.jsonl'
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
            output_path = os.path.join('../batch_id_database', txt_filename)
            with open(output_path, 'w') as f:
                for b_id in batch_ids:
                    f.write(b_id + "\n")
            # After submitting all batches, exit since we are not processing responses in real time
            exit(0)
            # === Edited Code Ends Here ===
        else:
            # import pdb; pdb.set_trace()
            with open(batch_result, 'r') as f:
                response_dict = json.load(f)
            # import pdb; pdb.set_trace()
            for idx in range(len(parsed_prompts)):
                try:
                    responses.append(response_dict[args.cache_name + f'_{sub_cache_name}_' + str(idx)])
                except KeyError:
                    print(f"KeyError: {args.cache_name + f'_{sub_cache_name}_' + str(idx)}")
                    responses.append("")
    output_costs = 0
    for r in responses:
        output_costs += get_output_price(r, args.llm)
    print(f"Output cost for {sub_cache_name} step", output_costs / ((not args.real_time) + 1))
    return responses

def match_string(sent, match_words):
    """
    Greedily matching words from `match_words` to `sent`.
    Parameters:
        sent (str): sentence string
        match_words (List[str]): list of words from sent, in the same order they appear in it.
    Returns:
        Optional[str]: string of length len(sent), for each symbol in sent, '^' if it contains in one
            of the match_words if aligned to sent, ' ' otherwise.
            Returns None if matching failed, e.g. due to words in match_words, which are not present
            in sent, or of the words are specified not in the same order they appear in the sentence.
    Example:
        sent = 'Lanny Flaherty is an American actor born on December 18, 1949, in Pensacola, Florida.'
        match_words = ['Lanny', 'Flaherty', 'born', 'on', 'December', '18', '1949']
        return '^^^^^ ^^^^^^^^                      ^^^^ ^^ ^^^^^^^^ ^^  ^^^^                        '
    """

    sent_pos = 0  # pointer to the sentence
    match_words_pos = 0  # pointer to the match_words list
    # Iteratively construct match_str with highlighted symbols, start with empty string
    match_str = ""
    while sent_pos < len(sent):
        # Check if current word cur_word can be located in sent[sent_pos:sent_pos + len(cur_word)]:
        # 1. check if symbols around word position are not letters
        # print(sent)
        # print(match_str)
        check_boundaries = False
        if sent_pos == 0 or not sent[sent_pos - 1].isalpha() or sent[sent_pos] == '-':
            check_boundaries = True
        # print("Sent pos", sent[sent_pos])
        # print("Sent pos - 1", sent[sent_pos - 1])
        # print(check_boundaries)
        if check_boundaries and match_words_pos < len(match_words):
            cur_match_word = match_words[match_words_pos]
            right_idx = sent_pos + len(cur_match_word)
            if right_idx < len(sent):
                check_boundaries = (not sent[right_idx].isalpha()) or (cur_match_word.endswith('-'))
            # 2. check if symbols in word position are the same as cur_word
            if check_boundaries and sent[sent_pos:].startswith(cur_match_word):
                # Found match at sent[sent_pos] with cur_word
                len_w = len(cur_match_word)
                sent_pos += len_w
                # Highlight this position in match string
                match_str += "^" * len_w
                match_words_pos += 1
                continue
        # No match at sent[sent_pos], continue with the next position
        sent_pos += 1
        match_str += " "

    if match_words_pos < len(match_words):
        # Didn't match all words to the sentence.
        # Possibly because the match words are in the wrong order or are not present in sentence.
        print(sent)
        print(match_words)
        print(match_str)
        return None

    return match_str


def extract_facts(args, output_texts):
    all_sentences = []
    sentence_idx = []
    sent_separator = ".?!。？！\n"
    for idx, text in enumerate(output_texts):
        if isinstance(text, str) and re.search(r'\w+', text):
            doc = nlp(text)
        else:
            continue
        for sent in doc.sentences:
            all_sentences.append(sent.text)
            sentence_idx.append(idx)
        if len(text) > 0 and text[-1] not in sent_separator:
            all_sentences = all_sentences[:-1]
            sentence_idx = sentence_idx[:-1]

    prompts = [EXTRACTION_PROMPT.format(sentence=sent) for sent in all_sentences]
    # import pdb; pdb.set_trace()
    responses = ask_open_ai(args, prompts, batch_result=args.extract_batch_result, sub_cache_name='extract')
    # import pdb; pdb.set_trace()
    all_claims = []
    claim_idx = []
    matching_prompts = []
    for idx, extracted_claims in enumerate(responses):
        if "EMPTY" in extracted_claims:
            continue
        for cid, claim_text in enumerate(extracted_claims.split("\n")):
            if not claim_text.startswith("- "):
                continue
            claim_text = claim_text[2:].strip()
            all_claims.append(claim_text)
            claim_idx.append((sentence_idx[idx], idx, cid))
            matching_prompt = MATCHING_PROMPT.format(
                sent=all_sentences[idx],
                claim=claim_text,
            )
            matching_prompts.append(matching_prompt)
    # import pdb; pdb.set_trace()
    match_responses = ask_open_ai(args, matching_prompts, batch_result=args.match_batch_result, sub_cache_name='match')

    matched_strings = []
    for i, match_words in enumerate(match_responses):
        match_words = match_words.strip().split(",")
        match_words = list(map(lambda x: x.strip(), match_words))
        parsed_match_words = []
        for w in match_words:
            w = w.strip("\"").strip("'")
            if len(w) == 0:
                continue
            elif ' ' not in w:
                parsed_match_words.append(w)
            else:
                w_split = [word for word in w.split(' ') if len(word) > 0]
                parsed_match_words.extend(w_split)
        sent = all_sentences[claim_idx[i][1]]
        matched_strings.append(match_string(sent, parsed_match_words))

    print(matched_strings.count(None) / len(matched_strings))

    result = {
        'all_sentences': all_sentences,
        'sentence_idx': sentence_idx,
        'all_claims': all_claims,
        'claim_idx': claim_idx,
        'matched_strings': matched_strings,
    }
    base_name = os.path.splitext(args.output_file)[0]  # Get the base name without the extension

    # Save as JSON
    with open(base_name + '.json', 'w') as f:
        json.dump(result, f)

    # Save as Pickle
    with open(base_name + '.pkl', 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_output", default=False, action='store_true')
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--prompt_column", type=str, default=None)
    parser.add_argument("--response_column", type=str, default=None)
    parser.add_argument("--cache_name", type=str, default="lima")
    parser.add_argument("--llm", type=str, default="gpt-4o", choices=['gpt-4o-mini', 'gpt-4o'])
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--real_time", default=False, action='store_true')
    parser.add_argument("--extract_batch_result", type=str, default=None)
    parser.add_argument("--match_batch_result", type=str, default=None)
    parser.add_argument("--has_reflection", default=False, action='store_true')
    parser.add_argument("--subset", default=False, action='store_true')
    parser.add_argument("--subset_size", type=int, default=10)
    args = parser.parse_args()

    if not args.llm_output:
        assert args.prompt_column and args.response_column
        if args.input_file.endswith('csv'):
            df = pd.read_csv(args.input_file)
        elif args.input_file.endswith('xlsx'):
            df = pd.read_excel(args.input_file)
        elif args.input_file.endswith('json'):
            df = pd.read_json(args.input_file)
        else:
            if args.input_file.endswith('jsonl'):
                lines = True
            else:
                lines = False
            print('here:',args.input_file)
            # import pdb; pdb.set_trace()
            df = pd.read_json(args.input_file, lines=lines)
        # import pdb; pdb.set_trace()
        responses = df[args.response_column].to_list()
    else:
        with open(args.input_file, 'r') as f:
            result_dict = json.load(f)
        responses = result_dict['greedy_texts']

    if args.sample > 0:
        responses = responses[:args.sample]

    reflection_count = 0
    if args.has_reflection:
        no_reflection =[]
        new_responses = []
        for i,r in enumerate(responses):
            assert len(r) == 1
            if r[0] is not None:
                if "<reflection>" in r[0]:
                    reflection_count +=1
                    new_responses.append(r[0].split("<reflection>")[0])
                else:
                    no_reflection.append(r[0])
                    new_responses.append(r[0])
            else:
                print("None response at index:", i)
                no_reflection.append(None)
                new_responses.append(None)
        
        print("Number of responses:", len(responses))
        print('Number of reflections:', reflection_count)
        responses = new_responses
    else:
        responses = [r[0] for r in responses]
        
        
    # import pdb; pdb.set_trace()

    if args.subset:
        responses = responses[:args.subset_size]

    # import pdb; pdb.set_trace()
    extract_facts(args, responses)





