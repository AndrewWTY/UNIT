import argparse
import json
from tqdm import tqdm
import pandas as pd
import stanza
import re
import os
import sys
import asyncio
from make_data_cut import create_answers_async, GENE_ARGS_DICT


EXTRACTION_PROMPT = """Break down the following sentence into atomic facts.
___
{sentence}
___

Respond with the following format:

- <atomic fact 1>
- <atomic fact 2>
...

However, if there is no factual claim, respond <EMPTY>."""


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='')
    parser.add_argument("--data", type=str, default='bio')
    parser.add_argument("--llm", type=str, default="gpt-4o", choices=['gpt-4o-mini', 'gpt-4o'])
    parser.add_argument("--cache_name", type=str, default='', help='Cache name for OpenAI batch API')
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--output_file", type=str, default='')
    parser.add_argument("--sample", type=int, default=-1)
    args = parser.parse_args()

    nlp = stanza.Pipeline(lang='en', processors='tokenize')
    df = pd.read_csv(args.input_file)
    user_questions = df['prompt'].tolist()
    raw_prompts = [p.split('<|user|>\n')[1].split('<|end_of_text|>')[0] for p in user_questions]
    if 'bio' in args.input_file:
        entities = [rp.split('a bio of ')[1].strip('.') for rp in raw_prompts]
    else:
        entities = [rp.split('you know about ')[1].strip('?') for rp in raw_prompts]

    all_sentences = []
    sentence_idx = []
    sent_separator = ".?!。？！\n"
    for idx, text in tqdm(enumerate(df['output']), total=len(df['output']), desc='Get prompts'):
        if "I am unconfident to precisely" in text:
            continue
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

    prompts = [[
                {'role': 'system', 'content': "You are a helpful AI assistant."},
                {'role': 'user', 'content': EXTRACTION_PROMPT.format(sentence=sent)},
            ] for sent in all_sentences]

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

    claims_per_sent = [[] for _ in range(len(all_sentences))]
    assert len(all_sentences) == len(responses)
    for idx, extracted_claims in tqdm(enumerate(responses), total=len(responses), desc="Parse responses"):
        if "EMPTY" in extracted_claims:
            continue
        for claim_text in extracted_claims.split("\n"):
            if not claim_text.startswith("- "):
                continue
            claim_text = claim_text[2:].strip()
            claims_per_sent[idx].append(claim_text)

    claims_per_answer = [[] for _ in range(len(df['output']))]
    assert len(sentence_idx) == len(claims_per_sent)
    for idx, sent_claims in tqdm(zip(sentence_idx, claims_per_sent), total=len(sentence_idx)):
        claims_per_answer[idx].extend(sent_claims)

    output_data = []
    for p, e, r, claims in tqdm(zip(raw_prompts, entities, df['output'], claims_per_answer), total=len(raw_prompts)):
        output_data.append({
            'prompt': p,
            'entity': e,
            'response': r,
            'unreflected_answer_claim': claims,
            'reflected_answer_claim': [],
        })

    with open(args.output_file, 'w') as f:
        for item in tqdm(output_data, total=len(output_data), desc="Write"):
            f.write(json.dumps(item))
            f.write('\n')


if __name__ == '__main__':
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
