import torch
import numpy as np
import string
import os
import json
import argparse
import pandas as pd
from torch.nn.functional import log_softmax

from typing import Dict, List
from dataclasses import dataclass
from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.stat_calculators import GreedyAlternativesNLICalculator, GreedyAlternativesFactPrefNLICalculator, StatCalculator
from lm_polygraph.estimators import MaximumClaimProbability, ClaimConditionedProbabilityClaim
from lm_polygraph.utils.deberta import Deberta
from lm_polygraph.stat_calculators.greedy_alternatives_nli import _eval_nli_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datetime import datetime
import time
import nltk
from typing import List
import math

nltk.download('stopwords')


def batchify(lst, batch_size):
    """Split the list `lst` into sublists of size `batch_size`."""
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


def batchify_claims(extraction_results, output_texts, batch_size):
    # import pdb; pdb.set_trace()
    all_sentences, sentence_idx, all_claims, claim_idx, matched_strings = extraction_results['all_sentences'], \
                                                                          extraction_results['sentence_idx'], \
                                                                          extraction_results['all_claims'], \
                                                                          extraction_results['claim_idx'], \
                                                                          extraction_results['matched_strings']

    claims_in_sentences = [[] for _ in range(len(all_sentences))]

    for claim_id, claim, matched_string in zip(claim_idx, all_claims, matched_strings):
        if matched_string is not None:
            claim_match_dict = {'claim': claim, 'matched_string': matched_string}
            claims_in_sentences[claim_id[1]].append(claim_match_dict)
    
    # import pdb; pdb.set_trace()

    sentences_in_outputs = [[] for _ in range(len(output_texts))] # len 440 which is the length of the original output
    claims_in_sentences_in_outputs = [[] for _ in range(len(output_texts))] # # len 440

    # sentence_idx , all_sentences, claims_in_sentences len 566 which is the length of the extracted fact
    # import pdb; pdb.set_trace()
    for i, (sent_id, sent, claim_list) in enumerate(zip(sentence_idx, all_sentences, claims_in_sentences)):
        sentences_in_outputs[sent_id].append(sent)
        claims_in_sentences_in_outputs[sent_id].append(claim_list)
    # import pdb; pdb.set_trace()

    batched_sentences_in_outputs = [sentences_in_outputs[i:i + batch_size] for i in range(0, len(sentences_in_outputs), batch_size)]
    batched_claims_in_sentences_in_outputs = [claims_in_sentences_in_outputs[i:i + batch_size] for i in range(0, len(claims_in_sentences_in_outputs), batch_size)]
    # import pdb; pdb.set_trace()
    return batched_sentences_in_outputs, batched_claims_in_sentences_in_outputs


class TargetProbsCalculator():
    """
    For Whitebox model (lm_polygraph.WhiteboxModel), at (input text, target text) batch calculates:
    * probabilities distribution of tokens in the generation target.
    """

    def __init__(self, n_alternatives: int = 10):
        self.n_alternatives = n_alternatives

    def __call__(
            self,
            input_texts: List[str],
            output_texts: List[str],
            model,
            tokenizer,
            **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the statistics of probabilities at each token position in the output text.

        Parameters:
            input_texts (List[str]): Input texts batch.
            output_texts (List[str]): Output texts batch.
            model (Model): Model used for generation.
        """
        if tokenizer.pad_token is None:
            existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                    len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        input_encodings = tokenizer(
            input_texts,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
            return_attention_mask=True,
        )
        # import pdb; pdb.set_trace()
        output_encodings = tokenizer(
            output_texts,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
            return_attention_mask=True,
            padding_side='right',
        )

        concat_ids = torch.cat([input_encodings.input_ids, output_encodings.input_ids], dim=1).to(model.device)
        attn_mask = torch.cat([input_encodings.attention_mask, output_encodings.attention_mask], dim=1).to(model.device)

        with torch.no_grad():
            logits = model(concat_ids, attention_mask=attn_mask).logits
            log_probs = log_softmax(logits, dim=-1).detach().cpu()
            del logits  # Important!
            torch.cuda.empty_cache()

        cut_log_probs = []
        cut_sequences = []
        alternatives = []
        log_likelihood = []
        for i in range(concat_ids.size(0)):
            start_idx = input_encodings.input_ids.shape[1]
            if tokenizer.pad_token_id in output_encodings.input_ids[i]:
                end_idx = (output_encodings.input_ids[i] == tokenizer.pad_token_id).nonzero()[0].item() + start_idx
            else:
                end_idx = len(concat_ids[i])
            sequence = concat_ids[i, start_idx:end_idx].cpu()
            cut_sequences.append(sequence.tolist())
            generated_log_probs = log_probs[i, start_idx - 1: end_idx - 1, :]
            cut_log_probs.append(log_probs[i, start_idx - 1: end_idx - 1, :].cpu().numpy())
            log_likelihood.append([generated_log_probs[j, sequence[j]] for j in range(len(generated_log_probs))])
            alternatives.append([[] for _ in range(generated_log_probs.size(0))])
            for j in range(generated_log_probs.size(0)):
                lt = generated_log_probs[j, :].cpu().numpy()
                best_tokens = np.argpartition(lt, - self.n_alternatives)
                ln = len(best_tokens)
                best_tokens = best_tokens[ln - self.n_alternatives: ln]
                label_token = sequence[j]
                if label_token not in best_tokens:
                    best_tokens = np.concatenate((best_tokens, [label_token]))
                for t in best_tokens:
                    alternatives[-1][j].append((t.item(), lt[t].item()))
                    alternatives[-1][j].sort(
                        key=lambda x: x[0] == sequence[j],
                        reverse=True,
                    )

            
            # Explicitly delete per-batch tensors after usage
            del generated_log_probs
            torch.cuda.empty_cache()

        result_dict = {
            "input_tokens": input_encodings["input_ids"].to("cpu").tolist(),
            "greedy_log_probs": cut_log_probs,
            "greedy_tokens": cut_sequences,
            "greedy_tokens_alternatives": alternatives,
            "greedy_texts": output_texts,
            "greedy_log_likelihoods": log_likelihood,
        }

        return result_dict


@dataclass
class Claim:
    claim_text: str
    # The sentence of the generation, from which the claim was extracted
    sentence: str
    # Indices in the original generation of the tokens, which are related to the current claim
    aligned_token_ids: List[int]


def align(
    sent: str,
    match_str: str,
    sent_tokens: List[int],
    tokenizer,
) -> List[int]:
    """
    Identifies token indices in `sent_tokens` that align with matching characters (marked by '^')
    in `match_str`. All tokens, which textual representations intersect with any of matching
    characters, are included. Partial intersections should be uncommon in practice.

    Args:
        sent: the original sentence.
        match_str: a string of the same length as `sent` where '^' characters indicate matches.
        sent_tokens: a list of token ids representing the tokenized version of `sent`.
        tokenizer: the tokenizer used to decode tokens.

    Returns:
        A list of integers representing the indices of tokens in `sent_tokens` that align with
        matching characters in `match_str`.
    """
    sent_pos = 0
    cur_token_i = 0
    # Iteratively find position of each new token.
    aligned_token_ids = []
    while sent_pos < len(sent) and cur_token_i < len(sent_tokens):
        cur_token_text = tokenizer.decode(sent_tokens[cur_token_i])
        # Try to find the position of cur_token_text in sentence, possibly in sent[sent_pos]
        if len(cur_token_text) == 0:
            # Skip non-informative token
            cur_token_i += 1
            continue
        if sent[sent_pos:].startswith(cur_token_text):
            # If the match string corresponding to the token contains matches, add to answer
            if any(
                t == "^"
                for t in match_str[sent_pos : sent_pos + len(cur_token_text)]
            ):
                aligned_token_ids.append(cur_token_i)
            cur_token_i += 1
            sent_pos += len(cur_token_text)
        else:
            # Continue with the same token and next position in the sentence.
            sent_pos += 1
    return aligned_token_ids


def align_facts(tokenizer, input_texts, output_texts, output_tokens, sentences_in_outputs, claims_in_sentences_in_outputs) -> Dict:
    """
    Given target output texts, and extracted facts from the output texts. Align the tokens of extracted facts to tokens in output texts.
    """

    ret_claims = []
    claim_texts_concatenated = []
    claim_input_texts_concatenated = []

    for output_id, (output_text, output_token, inp_text) in enumerate(zip(output_texts, output_tokens, input_texts)):
        sentences = sentences_in_outputs[output_id]
        claims_in_output = claims_in_sentences_in_outputs[output_id]


        sent_start_token_idx, sent_end_token_idx = 0, 0
        sent_start_idx, sent_end_idx = 0, 0
        claim_obj_in_output_text = []
        for s, claims_in_sent in zip(sentences, claims_in_output):
            # Find sentence location in text: text[sent_start_idx:sent_end_idx]

            assert s in output_text
            for start_position in range(len(output_text)):
                if output_text[start_position:].startswith(s):
                    sent_start_idx = start_position
                    break
            for end_position in range(1, len(output_text) + 1):
                if output_text[:end_position].endswith(s):
                    sent_end_idx = end_position
                    break

            # Iteratively decode tokenized text until decoded sequence length is
            # greater or equal to the starting position of current sentence.
            # Find sentence location in tokens: tokens[sent_start_token_idx:sent_end_token_idx]

            while sent_start_token_idx <= len(output_token) and len(tokenizer.decode(output_token[:sent_start_token_idx])) < sent_start_idx:
                sent_start_token_idx += 1
            while sent_end_token_idx <= len(output_token) and len(tokenizer.decode(output_token[:sent_end_token_idx])) < sent_end_idx:
                sent_end_token_idx += 1

            s_token = output_token[sent_start_token_idx:sent_end_token_idx]
            claim_obj_in_sent = []
            
            # import pdb; pdb.set_trace()
            for c in claims_in_sent:
                aligned_token_ids = align(s, c['matched_string'], s_token, tokenizer)
                if len(aligned_token_ids) == 0:
                    continue
                for idx in range(len(aligned_token_ids)):
                    aligned_token_ids[idx] += sent_start_token_idx
                claim_obj_in_sent.append(
                    Claim(claim_text=c['claim'], sentence=s, aligned_token_ids=aligned_token_ids)
                )
            claim_obj_in_output_text.extend(claim_obj_in_sent)

        ret_claims.append(claim_obj_in_output_text)
        for c in ret_claims[-1]:
            claim_texts_concatenated.append(c.claim_text)
            claim_input_texts_concatenated.append(inp_text)

    return {
        "claims": ret_claims,
        "claim_texts_concatenated": claim_texts_concatenated,
        "claim_input_texts_concatenated": claim_input_texts_concatenated,
    }


@dataclass
class DummyModel:
    tokenizer: AutoTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm_output", default=False, action='store_true')
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--cache_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--prompt_column", type=str, default=None)
    parser.add_argument("--response_column", type=str, default=None)
    parser.add_argument("--nli_context", type=str, default="fact_pref")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--claim_file", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--deberta_batch_size", type=int, default=16)
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--subset", default=False, action='store_true')
    args = parser.parse_args()

    if args.llm_output:
        with open(args.input_file, 'r') as f:
            result_dict = json.load(f)
        instruction_texts = result_dict['input_text']
        response_texts = result_dict['greedy_texts']
    else:
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
            df = pd.read_json(args.input_file, lines=lines)
        instruction_texts = df[args.prompt_column].to_list()
        response_texts = df[args.response_column].to_list()
    
    if args.subset:
        print("Subset")
        instruction_texts = instruction_texts[:10]
        response_texts = response_texts[:10]

    # # import pdb; pdb.set_trace()
    with open(args.claim_file, 'r') as f:
        extracted = json.load(f)

    index_batches = batchify(list(range(len(instruction_texts))), batch_size=args.batch_size)
    input_batches = batchify(instruction_texts, batch_size=args.batch_size)
    output_batches = batchify(response_texts, batch_size=args.batch_size)
    sent_in_output_batches, claim_in_sent_in_output_batches = batchify_claims(extracted,
                                                                              response_texts, batch_size=args.batch_size)
    # import pdb; pdb.set_trace()
    if args.sample > 0:
        input_batches = input_batches[:args.sample]
        output_batches = output_batches[:args.sample]
        sent_in_output_batches = sent_in_output_batches[:args.sample]
        claim_in_sent_in_output_batches = claim_in_sent_in_output_batches[:args.sample]

    assert len(input_batches) == len(sent_in_output_batches) \
           and len(input_batches) == len(claim_in_sent_in_output_batches)

    if not args.llm_output:
        # raw_model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir=args.cache_dir, device_map='auto')
        # raw_model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='auto')
        if '14b' in args.model_name.lower():    
            print("Using float16")
            raw_model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='auto', torch_dtype=torch.float16)
        else:
            print("Using float32")
            raw_model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='auto')

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = WhiteboxModel(model=raw_model, tokenizer=tokenizer, model_path=args.model_name)

        if 'llama-3' in args.model_name.lower():
            tokenizer.pad_token = "<|reserved_special_token_0|>"
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<|reserved_special_token_0|>')
        probs_cal = TargetProbsCalculator(n_alternatives=10)
    else:
        raw_model = None
        probs_cal = None
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name)
        model = DummyModel(tokenizer=tokenizer)

    if os.path.exists(args.cache_file):
        with open(args.cache_file, 'r') as f:
            all_results = json.load(f)
    else:
        print("No cache file found")
        all_results = {}

    if args.nli_context == "no_context":
        nli_cal = GreedyAlternativesNLICalculator(Deberta(batch_size=args.deberta_batch_size))
    else:
        nli_cal = GreedyAlternativesFactPrefNLICalculator(Deberta(batch_size=args.deberta_batch_size))
    # import pdb; pdb.set_trace()
    for batch_idx, (indices, raw_input_batch, raw_output_batch, raw_sent_in_output, raw_claim_in_sent_in_output) \
            in tqdm(enumerate(zip(index_batches, input_batches, output_batches,
                                  sent_in_output_batches, claim_in_sent_in_output_batches)), total=len(input_batches)):
        if str(batch_idx) in all_results.keys(): # PROBLEM IS HERE
            # import pdb; pdb.set_trace()
            print("Skip Batch", batch_idx)
            continue

        input_batch = []
        output_batch = []
        sent_in_output = []
        claim_in_sent_in_output = []
        for i, o, s, c in zip(raw_input_batch, raw_output_batch, raw_sent_in_output, raw_claim_in_sent_in_output):
            # Check if o is a non-empty string
            if isinstance(o, str) and len(o) > 0:
                input_batch.append(i)
                output_batch.append(o)
                sent_in_output.append(s)
                claim_in_sent_in_output.append(c)
            # Check if o is a single-element list
            elif isinstance(o, list) and len(o) == 1:
                # Check if that single element is NaN
                if isinstance(o[0], float) and math.isnan(o[0]):
                    print("Correcting entry because o[0] is NaN:", o)
                    o[0] = [""]
                # Convert to string if necessary
                input_batch.append(i)
                output_batch.append(str(o[0]))
                sent_in_output.append(s)
                claim_in_sent_in_output.append(c)
            else:
                print("Invalid output encountered and skipped:")
                print("i:", i)
                print("o:", o)
                print("s:", s)
                print("c:", c)

        # import pdb; pdb.set_trace()
        print(datetime.now().time(), "Batch", batch_idx, "Start Compute Logits")
        if not args.llm_output:

            stats = probs_cal(input_texts=input_batch, output_texts=output_batch, model=raw_model, tokenizer=tokenizer)
        else:
            stats = {}
            for key, values in result_dict.items():
                batch_values = [v for v_id, v in enumerate(values) if v_id in indices]
                stats[key] = batch_values
        print(datetime.now().time(), "Batch", batch_idx, "Finish Compute Logits")
        greedy_texts = stats['greedy_texts']
        greedy_tokens = stats['greedy_tokens']
        all_claims = align_facts(tokenizer, input_batch, greedy_texts, greedy_tokens,
                                 sent_in_output, claim_in_sent_in_output)
        stats.update(all_claims)

        max_prob = MaximumClaimProbability()
        max_prob_results = max_prob(stats)

        print(datetime.now().time(), "Batch", batch_idx, "Start Compute NLI")
        stats.update(nli_cal(stats, input_batch, model))
        print(datetime.now().time(), "Batch", batch_idx, "Finish Compute NLI")

        ccp = ClaimConditionedProbabilityClaim(nli_context=args.nli_context)
        ccp_results = ccp(stats)

        batch_results = []
        for i in range(len(output_batch)):
            # Build a list of claim-level dictionaries
            result_in_i_th = []
            for claim, max_prob, ccp in zip(all_claims['claims'][i],
                                            max_prob_results[i],
                                            ccp_results[i]):
                result_in_i_th.append({
                    'claim': claim.claim_text,
                    'max_prob': float(max_prob),
                    'ccp': float(ccp),
                })

            # Attach raw_claim_in_sent_in_output[i] at the same level, outside of "claims"
            batch_results.append({
                'claims': result_in_i_th,
                'raw_claim_in_sent_in_output': raw_claim_in_sent_in_output[i],
            })

        # Then store in all_results
        all_results[str(batch_idx)] = batch_results

        with open(args.cache_file, 'w') as f:
            json.dump(all_results, f)

    # import pdb; pdb.set_trace()
    final_result = []
    for i in range(len(all_results.keys())):
        final_result.extend(all_results[str(i)])
    
    # import pdb; pdb.set_trace()

    assert len(instruction_texts) == len(response_texts) == len(final_result)

    parsed_result = []
    for re, instruction, response in zip(final_result, instruction_texts, response_texts):
        # import pdb; pdb.set_trace()
        parsed_result.append({
            'instruction': instruction,
            'response': response,
            # The "claim_uncertainty" key can be the content of "re['claims']"
            'claim_uncertainty': re['claims'],
            # Then attach the raw claim data outside "claim_uncertainty"
            'raw_claim_in_sent_in_output': re['raw_claim_in_sent_in_output'],
        })

    import pprint

    with open(args.output_file, 'w') as f:
        for p in parsed_result:
            f.write(json.dumps(p))
            f.write('\n')





