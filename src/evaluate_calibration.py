import pandas as pd
import numpy as np
import re
import json
import ast
from tqdm import tqdm
import argparse
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from openai import OpenAI
import time
import os
import pandas as pd
import json
import bisect

os.environ['HF_TOKEN']="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"
# os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "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"
class GPT_API_Caller:

    def __init__(self, model):
        self.model = model
        self.gpt = OpenAI()

    def __call__(self, system_prompt, user_prompt, max_tokens = 300, temperature = 0.7, response_format = { "type": "text" }):

        for _ in range(3):
            try:
                response = self.gpt.chat.completions.create(
                      model=self.model,
                      messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                      ],
                      temperature = temperature,
                      max_tokens = max_tokens,
                      response_format = response_format
                      )
                content = response.choices[0].message.content
                break

                if response.choices[0].finish_reason == 'length':
                  max_tokens += 200
                  raise Exception(f"Stop at max_token, current max_tokens is {max_tokens}, before is {max_tokens-200}")

            except Exception as e:
                print(e)
                time.sleep(1)

        return content



SORTING_PROMPT = '''\
### Instruction
You will be given a question and two list relating to the question, claim list and reflection list that was extracted from an answer to the question.
Please help to extract two new list from the claim list and the reflection list:
1. Covered Claims: All the claims in Claim list that is COVERED by at least one of the reflections in reflection list.
2. Covered Reflection: All the reflections in reflection list that is COVERED by at least one of the claims in Claim list.

For Example:
- Question:
Tell me a bio of Cheyenne Brando.

- Claim List:
Cheyenne Brando was born in 1996.
Cheyenne Brando is the daughter of Marlon Brando.
Cheyenne Brando is the daughter of Tarita Teriipaia.
She was born in Tahiti.
Her parents lived in Tahiti after they married.
Her parents married following the filming of Mutiny on the Bounty.
She has a half-sister named Miko.
Miko is from Brando's relationship with his second wife.
Brando's second wife is Movita Castaneda.
Cheyenne Brando is named after a character.
Cheyenne Brando's father has a character in The Wild One.

- Reflection List:
Marlon Brando was an actor.
Marlon Brando had a relationship with Movita Castaneda.
Miko is a half-sister of Cheyenne Brando.
Cheyenne Brando is named after her father's character in The Wild One.

# Output
- Covered Claims:
She has a half-sister named Miko.
Brando's second wife is Movita Castaneda.
Cheyenne Brando is named after a character.
Cheyenne Brando's father has a character in The Wild One.

- Covered Reflection:
Marlon Brando had a relationship with Movita Castaneda.
Miko is a half-sister of Cheyenne Brando.
Cheyenne Brando is named after her father's character in The Wild One.

Now it's your turn to answer, follow the format in the example strictly:
- Question:
{0}

- Claim List:
{1}

- Reflection List:
{2}
'''

def load_jsonl(file_path):
    """Loads a JSONL file into a list of dictionaries.

    Args:
        file_path: The path to the JSONL file.

    Returns:
        A list of dictionaries, where each dictionary represents a JSON object
        from the file. Returns an empty list if the file is empty or does not exist.
    """
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON object: {e}")
                    print(f"Problematic line: {line}")
        if not data:
            print(f"No data found in the file: {file_path}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    return data
    
def extract_entity(text):
    """
    Extracts the question from a given text formatted with <|user|> and <|end_of_text|> tags.

    Parameters:
        text (str): The input text containing the question.

    Returns:
        str: The extracted question, or None if not found.
    """
    match = re.search(r'<\|start_header_id\|>user<\|end_header_id\|>\n\nQuestion: Tell me a bio of(\s*.*?).<', text, re.DOTALL)
    match2= re.search(r'<\|start_header_id\|>user<\|end_header_id\|>\n\nIn a paragraph, could you tell me what you know about(\s*.*?)<', text, re.DOTALL)
    match3 = re.search(r'Question: Tell me a bio of(\s*.*?).<', text, re.DOTALL)
    match4 = re.search(r'In a paragraph, could you tell me what you know about(\s*.*?)<', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    elif match2:
        return match2.group(1).strip()
    elif match3:
        return match3.group(1).strip()
    elif match4:
        return match4.group(1).strip()
    else:
      raise Exception('No entity found')

def extract_question(text):
    """
    Extracts the question from a given text formatted with <|user|> and <|end_of_text|> tags.

    Parameters:
        text (str): The input text containing the question.

    Returns:
        str: The extracted question, or None if not found.
    """
    match = re.search(r'Question:\s*(.+?)(?=[\n\r<])', text)
    match2 = re.search(r'<\|start_header_id\|>user<\|end_header_id\|>\n\n(In a paragraph, could you tell me what you know about.*?)<', text, re.DOTALL)
    match3 = re.search(r'(?<=<\|user\|\>\n)([\s\S]*?)(?=<\|end_of_text\|\>)', text, re.DOTALL)
    match4 = re.search(r'Question:\s*(.+?)(?=[\n\r<])', text)
    match5 = re.search(r'In a paragraph, could you tell me what you know about .*?\?', text)

    # import pdb; pdb.set_trace()
    if match:
        return match.group(0).strip()
    elif match2:
        return match2.group(1).strip()
    elif match3:
        return match3.group(1).strip()
    elif match4:
        return match4.group(0).strip()
    elif match5:
        return match5.group(0).strip()
    else:
        raise Exception('No question found')

def extract_reflections(response_text):
    """
    Extracts reflections from the response text after the <reflection> token.
    Strips numbering from each reflection.
    """
    unconfident_text = '<reflection>I am unconfident about the accuracy and the truthfulness of most of the information provided above.'
    confident_text = '<reflection>I am confident about the accuracy and the truthfulness of the information provided.'
    if unconfident_text in response_text:
        return [], 'unconfident'
    elif confident_text in response_text:
        return [], 'confident'

    reflection_marker = '<reflection>'
    reflection_start = response_text.find(reflection_marker)
    if reflection_start == -1:
        return [], None

    reflection_text = response_text[reflection_start + len(reflection_marker):]
    reflection_text = reflection_text.strip()

    pattern = r'\d+\.\s*(.*?)(?=\n\d+\. |\Z)'
    reflections = re.findall(pattern, reflection_text, re.DOTALL)
    reflections = [ref.strip() for ref in reflections]

    return reflections, None

def extract_sorting_response(input_text):
    lines = input_text.strip().split("\n")
    covered_claims = []
    covered_reflection = []
    current_section = None

    for line in lines:
        line = line.strip()
        if line.startswith("- Covered Claims:"):
            current_section = "claims"
        elif line.startswith("- Covered Reflection:"):
            current_section = "reflection"
        elif line:
            if current_section == "claims":
                covered_claims.append(line)
            elif current_section == "reflection":
                covered_reflection.append(line)

    return {
        "Covered Claims": covered_claims,
        "Covered Reflection": covered_reflection
    }

def extract_claims_ccp(claim_uncertainty_text):
    try:
        claims = [item['claim'] for item in claim_uncertainty_text]
        ccp_values = [float(item['ccp']) for item in claim_uncertainty_text]
        return claims, ccp_values
    except (ValueError, SyntaxError, TypeError, KeyError):
        return [], []

def parse_claim(row):
    raw_claim = row['raw_claim_in_sent_in_output']
    result = []
    for i, r in enumerate(raw_claim):
      result.extend(r)
    result = [d['claim'] for d in result]
    return result

def find_quantile(x, data):
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 0:
        raise ValueError("Data list must not be empty.")

    if x <= sorted_data[0]:
        return 0.0
    if x >= sorted_data[-1]:
        return 1.0

    i = bisect.bisect_left(sorted_data, x)
    if sorted_data[i] == x:
        return (i + 1) / n

    x_low  = sorted_data[i - 1]
    x_high = sorted_data[i]
    frac_low  = i / n         # fraction of data ≤ sorted_data[i-1]
    frac_high = (i + 1) / n   # fraction of data ≤ sorted_data[i]
    proportion = (x - x_low) / (x_high - x_low)

    return frac_low + proportion * (frac_high - frac_low)
    
def calculate_mean_error(threshold, candidates):
    
    if not (-1 <= threshold <= 0):
        raise ValueError("Threshold must be in the range [-1, 0].")

    for candidate in candidates:
        if candidate > threshold: 
            raise ValueError(f"Candidate number {candidate} is greater than the threshold {threshold}.")

    if not candidates:
        return -np.inf

    mean_error = sum(threshold - candidate for candidate in candidates) / len(candidates)
    return mean_error

def compute_ranked_error_four_lists(incorrect_in_reflection, correct_in_reflection, incorrect_in_unreflection, correct_in_unreflection, threshold):
    combined_all = (
        incorrect_in_reflection
        + incorrect_in_unreflection
        + correct_in_reflection
        + correct_in_unreflection
        + [threshold]
    )

    sorted_values = sorted(combined_all)
    threshold_rank = sorted_values.index(threshold)

    combined_incorrect = incorrect_in_reflection + incorrect_in_unreflection
    combined_correct   = correct_in_reflection + correct_in_unreflection

    def get_ranks(values, sorted_list):
        return [sorted_list.index(v) for v in values]

    def ranked_error_for_group(values):
        ranks = get_ranks(values, sorted_values)
        diffs = [abs(r - threshold_rank) for r in ranks]
        return sum(diffs) / len(diffs) if diffs else 0.0

    ranked_error_incorrect = ranked_error_for_group(combined_incorrect)
    ranked_error_correct   = ranked_error_for_group(combined_correct)

    return ranked_error_incorrect, ranked_error_correct

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process reflections and claims.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSONL file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output CSV file.')
    parser.add_argument('--ccp_threshold', type=float, default=-0.5, help='CCP threshold for selecting uncertain claims.')
    parser.add_argument('--match_threshold', type=float, default=0.5, help='Word overlap threshold for matching claims and reflections.')
    parser.add_argument('--grid_search', default = False, action='store_true', help='Perform grid search to find the best CCP threshold.')
    args = parser.parse_args()
    print('Using CCP threshold:', args.ccp_threshold)

    partial_output_file = args.output_file.replace('.jsonl','_partial.jsonl').replace('calibration_result/','calibration_result/partial/')
    # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    if os.path.exists(partial_output_file):
        print(f"Output file {partial_output_file} already exists. Skipping processing answer claims.")
        df = pd.read_json(partial_output_file, lines=True)
        # import pdb; pdb.set_trace()
            # try:
        df['entity'] = df['instruction'].apply(extract_entity)
        # df['prompt'] = output_df_without_base['prompt']
    elif 'base' in args.input_file and os.path.exists(partial_output_file.replace('_base','')):
        print(f"Output file {partial_output_file} already exists. Now processing Result on base model")
        output_file_without_base = partial_output_file.replace('_base','')
        output_df_without_base = pd.read_json(output_file_without_base, lines=True)
        # import pdb; pdb.set_trace()
        df = pd.DataFrame(load_jsonl(args.input_file))
        # import pdb; pdb.set_trace()
        df['sorting_response'] = output_df_without_base['sorting_response']

        try:
            df['entity'] = df['instruction'].apply(extract_entity)
        except Exception as e:
            print(f"Error extracting entity {e}")
            
        df['prompt'] = output_df_without_base['prompt']
        df['answer_claims'] = output_df_without_base['answer_claims']
        df['reflection_claims'] = output_df_without_base['reflection_claims']
        df['confident_status'] = output_df_without_base['confident_status']
        # import pdb; pdb.set_trace()
    else:
        print(f"Processing input file {args.input_file}...")
        df = pd.DataFrame(load_jsonl(args.input_file))
        
        if df.empty:
            print("No data to process.")
            return

        df['response'] = df['response'].apply(lambda x: x[0] if isinstance(x, list) and len(x) == 1 else x)
        # import pdb; pdb.set_trace()
        df['entity'] = df['instruction'].apply(extract_entity)
        df['prompt'] = df['instruction'].apply(extract_question)
        # import pdb; pdb.set_trace()
        df['answer_claims'] = df.apply(parse_claim, axis=1)
        # import pdb; pdb.set_trace()
        # ccp_threshold = args.ccp_threshold

        results = []
        empty_reflection_count = 0  

        df['reflection_claims'] = None
        df['confident_status'] = None
        gpt_caller = GPT_API_Caller('gpt-4o')

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Extracting reflections_claims, and confident_status"):
            response_text = row['response']
            claim_uncertainty_text = row['claim_uncertainty']

            # Extract reflections
            if isinstance(response_text, list) and len(response_text) == 1:
                response_text = response_text[0]

            reflections, confident_status = extract_reflections(response_text)

            if len(reflections) == 0:
                empty_reflection_count += 1

            df.at[index, 'reflection_claims'] = reflections
            df.at[index, 'confident_status'] = confident_status

        count=0
        df['sorting_response'] = None
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating sorting responses for all answer claims"):

            if df.at[index, 'answer_claims'] == [] or df.at[index, 'reflection_claims'] == []:
                pass
            else:
                # import pdb; pdb.set_trace()
                prompt_claims = "\n".join(row['answer_claims'])
                prompt_reflections = "\n".join(row['reflection_claims'])
                prompt_prompt = row['prompt']
                user_prompt = SORTING_PROMPT.format(prompt_prompt, prompt_claims, prompt_reflections)

                responses = gpt_caller(system_prompt="You are a helpful assistant.", user_prompt=user_prompt, temperature = 0, response_format = { "type": "text" })
                df.at[index, 'sorting_response'] = responses
                count+=1
    

        print(f'answer claim and reflection claim both not empty: {count}')
        count=0
        subset1 = df.copy()
        subset1.to_json(partial_output_file, orient='records', lines=True)


    ccp_threshold_list = np.arange(-1, 0, 0.0005)
    # ccp_threshold_list = np.arange(-1, 0, 0.01)
    best_ccp_threshold_wo_p = 0
    best_macro_precision_wo_unconfident_wo_p = 0
    best_macro_precision_all_wo_p = 0
    best_macro_precision_wo_all_wo_p = 0
    best_macro_recall_wo_p = 0
    best_macro_specificity_wo_p = 0
    best_macro_rac_ac_wo_p = 0
    best_micro_rac_ac_wo_p = 0
    # best_mean_error_wo_p = 1
    best_rac_ccp_wo_p = 0
    best_mean_error_wo_p = 0
    best_count1 = 0
    best_count2 = 0
    best_inc_error = 0
    best_cor_error = 0
    best_macro_incor_error_wo_p = 0
    best_macro_cor_error_wo_p = 0
    best_balanced_accuracy_wo_p = 0
    
    best_average_reflection_count = 0
    best_unconfident_correct_count_wo_p = 0
    best_unconfident_count_wo_p = 0
    best_precision_zero_div_count_wo_p = 0
    best_recall_zero_div_count_wo_p = 0
    best_specificity_zero_div_count_wo_p = 0

    if args.grid_search:
        ccp_threshold_list = list(ccp_threshold_list)+[args.ccp_threshold]
    else:
        ccp_threshold_list = [args.ccp_threshold]
    # import pdb; pdb.set_trace()

    for ccp_threshold in tqdm(ccp_threshold_list, desc="Finding the Model Learnt CCP Threshold"):
        df['reflected_answer_claim'] = None # unconfident: answer_claim , confident: [], unconfident: reflection=answer_claims, confident: empty reflection
        df['covered_reflection'] = None # unconfident: answer_claim , confident: [], unconfident: since reflection=answer_claims, confident: empty reflection
        df['unreflected_answer_claim'] = None # unconfident: [], confident: answer_claim, unconfident: since reflection=answer_claims, confident: empty reflection so every answer_claim is not reflected
        df['uncovered_reflection'] = None # unconfident: [], confident: [], unconfident: since reflection = answer_claims, confident: empty reflection

        # unconfident: reflection_claims = answer_claims,  confident = reflection_claims = []
        df['reflected_uncertain_claim'] = None # unconfident: uncertain_claim , confident: [], unconfident: reflection=answer_claims, uncertain_claims is a subset of answer_claims, all uncertain_claims are reflected. confident: empty reflection
        df['unreflected_uncertain_claim'] = None # unconfident: [], confident: uncertain_claims, unconfident: reflection=answer_claims, uncertain_claims is a subset of answer_claims, all uncertain_claims are reflected. confident: no reflection so all uncertain claim is unreflected
   
        # create uncertain_claims based on ccp_threshold
        df['uncertain_claims'] = None
        for index, row in df.iterrows():
        # for index, row in tqdm(df.iterrows(), total=len(df), desc="Extracting reflections_claims, and confident_status"):
            claim_uncertainty_text = row['claim_uncertainty']
            claims, ccp_values = extract_claims_ccp(claim_uncertainty_text)
            uncertain_claims = [claim for claim, ccp in zip(claims, ccp_values) if ccp >= ccp_threshold]
            df.at[index, 'uncertain_claims'] = uncertain_claims

        for index, row in df.iterrows():
        # for index, row in tqdm(df.iterrows(), total=len(df), desc="Removing duplicates"):
            df.at[index, 'answer_claims'] = list(set(row['answer_claims']))
            df.at[index, 'uncertain_claims'] = list(set(row['uncertain_claims']))
            df.at[index, 'reflection_claims'] = list(set(row['reflection_claims']))

        for index, row in df.iterrows():
        # for index, row in tqdm(df.iterrows(), total=len(df), desc="Edge Case Handling"):
            if row['confident_status'] == 'unconfident':
                df.at[index,'reflection_claims'] = row['answer_claims']

                df.at[index,'reflected_answer_claim'] = row['answer_claims'] 
                df.at[index,'covered_reflection'] = row['answer_claims']
                df.at[index,'unreflected_answer_claim'] = []
                df.at[index,'uncovered_reflection'] = []

                df.at[index,'reflected_uncertain_claim'] = row['uncertain_claims']
                df.at[index,'unreflected_uncertain_claim'] = [] 

            elif row['confident_status'] == 'confident':
                df.at[index,'reflection_claims'] = []
                
                df.at[index,'reflected_answer_claim'] = [] 
                df.at[index,'covered_reflection'] = []
                df.at[index,'unreflected_answer_claim'] = row['answer_claims']
                df.at[index,'uncovered_reflection'] = []

                df.at[index,'reflected_uncertain_claim'] = [] 
                df.at[index,'unreflected_uncertain_claim'] = row['uncertain_claims'] 

            elif pd.isnull(row['confident_status']):
                if row['uncertain_claims']==[]:
                    df.at[index,'reflected_uncertain_claim'] = [] 
                    df.at[index,'unreflected_uncertain_claim'] = []

                if row['answer_claims']==[]:
                    df.at[index,'reflected_answer_claim'] = []
                    df.at[index,'covered_reflection'] = []
                    df.at[index,'unreflected_answer_claim'] = []
                    df.at[index,'uncovered_reflection'] = row['reflection_claims']

                if row['reflection_claims']==[]:
                    df.at[index,'reflected_answer_claim'] = []
                    df.at[index,'covered_reflection'] = []
                    df.at[index,'unreflected_answer_claim'] = row['answer_claims']
                    df.at[index,'uncovered_reflection'] = []

                    df.at[index,'reflected_uncertain_claim'] = []
                    df.at[index,'unreflected_uncertain_claim'] = row['uncertain_claims']
                else:
                    pass

            else:
                pass
        
        for index, row in df.iterrows():
            if df.at[index, 'answer_claims'] == [] or df.at[index, 'reflection_claims'] == [] or not pd.isnull(df.at[index, 'confident_status']): 
                continue
            else:
                try:
                    df.at[index, 'reflected_answer_claim'] = extract_sorting_response(row['sorting_response'])['Covered Claims']
                    df.at[index, 'covered_reflection'] = extract_sorting_response(row['sorting_response'])['Covered Reflection']
                    df.at[index, 'unreflected_answer_claim'] = [claim for claim in row['answer_claims'] if claim not in df.at[index, 'reflected_answer_claim']]
                    df.at[index, 'uncovered_reflection'] = [reflection for reflection in row['reflection_claims'] if reflection not in df.at[index, 'covered_reflection']]
                except:
                    print(row['sorting_response'])
                    import pdb; pdb.set_trace()

        for index, row in df.iterrows():
            if df.at[index, 'uncertain_claims'] == [] or df.at[index, 'reflection_claims'] == [] or not pd.isnull(df.at[index, 'confident_status']): 
                continue
            else:
                df.at[index, 'reflected_uncertain_claim'] = [claim for claim in row['uncertain_claims'] if claim in df.at[index, 'reflected_answer_claim']]
                df.at[index, 'unreflected_uncertain_claim'] = [claim for claim in row['uncertain_claims'] if claim not in df.at[index, 'reflected_uncertain_claim']]



        df['precision'] = None
        df['recall'] = None
        df['specificity'] = None
        df['uncertain_correct'] =None
        df['rac/ac'] = None
        df['reflection_count'] = df['reflection_claims'].apply(len)
        df['incor_ccp_deviation'] = None
        df['cor_ccp_deviation'] = None
        
        precision_zero_div_count = 0
        recall_zero_div_count = 0
        specificity_zero_div_count = 0
        extreme_error_1 = 0
        extreme_error_2 = 0
    
        for index, row in df.iterrows():
        # for index, row in tqdm(df.iterrows(), total=len(df), desc="Calculating metrics"):
            # Precision
            try:
                if len(row['reflected_answer_claim']) == 0:
                    precision_zero_div_count += 1
                    df.at[index, 'precision'] = None
                else:
                    df.at[index, 'precision'] = len(row['reflected_uncertain_claim']) / len(row['reflected_answer_claim'])  # TP/(TP+FP)
                    if df.at[index, 'precision'] < 0:
                        print(index)
                        import pdb; pdb.set_trace()

                # Recall
                if len(row['uncertain_claims']) == 0:
                    recall_zero_div_count += 1
                    df.at[index, 'recall'] = None
                else:
                    df.at[index, 'recall'] = len(row['reflected_uncertain_claim']) / len(row['uncertain_claims'])  # TP/(TP+FN)
                    if df.at[index, 'recall'] < 0:
                        print(index)
                        import pdb; pdb.set_trace()

                
                # Specificity
                if len(row['answer_claims']) - len(row['uncertain_claims']) == 0:
                    specificity_zero_div_count += 1
                    df.at[index, 'specificity'] = None
                else:
                    df.at[index, 'specificity'] = len([claim for claim in row['unreflected_answer_claim'] if claim not in row['uncertain_claims']]) / (len(row['answer_claims']) - len(row['uncertain_claims'])) 
                    if df.at[index, 'specificity'] < 0:
                        print(index)
                        import pdb; pdb.set_trace()

                # Uncertain Correct
                if len(row['uncertain_claims']) >= 10 and row['confident_status'] == 'unconfident':
                    df.at[index, 'uncertain_correct'] = True

                # rac/ac
                if len(row['answer_claims']) == 0:
                    df.at[index, 'rac/ac'] = None
                else:
                    df.at[index, 'rac/ac'] = len(row['reflected_answer_claim']) / len(row['answer_claims'])

                # calculate ccp deviation per row
                reflected_answer_claim_ccp = []
                unreflected_answer_claim_ccp = []
                claim_dict = {item['claim']: item['ccp'] for item in row['claim_uncertainty']}

                for c in row['reflected_answer_claim']:
                    if c in claim_dict:
                        reflected_answer_claim_ccp.append(claim_dict[c])
                for c in row['unreflected_answer_claim']:
                    if c in claim_dict:
                        unreflected_answer_claim_ccp.append(claim_dict[c])

                inc_in_reflection = [ccp for ccp in reflected_answer_claim_ccp if ccp <= ccp_threshold]
                cor_in_reflection = [ccp for ccp in reflected_answer_claim_ccp if ccp > ccp_threshold]
                inc_in_unreflection = [ccp for ccp in unreflected_answer_claim_ccp if ccp > ccp_threshold]
                cor_in_unreflection = [ccp for ccp in unreflected_answer_claim_ccp if ccp <= ccp_threshold]

                inc_error, cor_error = compute_ranked_error_four_lists(
                    inc_in_reflection,
                    cor_in_reflection,
                    inc_in_unreflection,
                    cor_in_unreflection,
                    ccp_threshold
                )

                df.at[index, 'incor_ccp_deviation'] = inc_error
                df.at[index, 'cor_ccp_deviation'] = cor_error
                

            except Exception as e:
                print(e)
                print(index)
                print(row)
                import pdb; pdb.set_trace()


        macro_precision_wo_unconfident = np.mean([
            row['precision']
            for _, row in df.iterrows()
            if not pd.isnull(row['precision']) and row['confident_status'] != 'unconfident'
        ])

        macro_precision_all = np.mean([
            row['precision']
            for _, row in df.iterrows()
            if not pd.isnull(row['precision']) 
        ])

        macro_precision_wo_all = np.mean([
            row['precision']
            for _, row in df.iterrows()
            if not pd.isnull(row['precision']) and pd.isnull(row['confident_status'])
        ])

        unconfident_correct_count = 0
        unconfident_count = 0
        for _, row in df.iterrows():
            if row['uncertain_correct']:
                unconfident_correct_count += 1
            if row['confident_status'] == 'unconfident':
                unconfident_count += 1
        

        macro_recall = np.mean([value for value in df['recall'] if not pd.isnull(value)])
        macro_specificity = np.mean([value for value in df['specificity'] if not pd.isnull(value)])
        macro_rac_ac = np.mean([value for value in df['rac/ac'] if not pd.isnull(value)])
        micro_rac_ac = len(df['reflected_answer_claim'].sum())/len(df['answer_claims'].sum())
        average_reflection_count = np.mean([value for value in df['reflection_count'] if not pd.isnull(value)])
        macro_inc_error = np.mean([value for value in df['incor_ccp_deviation']])
        macro_cor_error = np.mean([value for value in df['cor_ccp_deviation']])

        ## calculate mean error
        reflected_answer_claim_ccp = []
        unreflected_answer_claim_ccp = []
        # num_rcc = 0
        count1 = 0
        count2 = 0

        for idx, row in df.iterrows():
            claim_dict = {item['claim']: item['ccp'] for item in row['claim_uncertainty']}

            for c in row['reflected_answer_claim']:
                if c in claim_dict:
                    reflected_answer_claim_ccp.append(claim_dict[c])
                else:
                    count1+=1
            
            for c in row['unreflected_answer_claim']:
                if c in claim_dict:
                    unreflected_answer_claim_ccp.append(claim_dict[c])
                else:
                    count2+=1

        avg_rac_ccp = abs(np.mean(reflected_answer_claim_ccp))
        avg_urac_ccp = abs(np.mean(unreflected_answer_claim_ccp))

        # import pdb; pdb.set_trace()

        incorrect_in_reflection = [ccp for ccp in reflected_answer_claim_ccp if ccp <= args.ccp_threshold]
        correct_in_reflection = [ccp for ccp in reflected_answer_claim_ccp if ccp > args.ccp_threshold]
        incorrect_in_unreflection = [ccp for ccp in unreflected_answer_claim_ccp if ccp > args.ccp_threshold]
        correct_in_unreflection = [ccp for ccp in unreflected_answer_claim_ccp if ccp <= args.ccp_threshold]
        # Compute the ranked errors
        inc_error, cor_error = compute_ranked_error_four_lists(
            incorrect_in_reflection,
            correct_in_reflection,
            incorrect_in_unreflection,
            correct_in_unreflection,
            args.ccp_threshold
        )

        macro_balanced_accuracy = (macro_recall + macro_specificity) / 2
        if macro_recall > best_macro_recall_wo_p:
            # import pdb; pdb.set_trace()
            best_macro_precision_wo_unconfident_wo_p = macro_precision_wo_unconfident
            best_macro_precision_all_wo_p = macro_precision_all
            best_macro_precision_wo_all_wo_p = macro_precision_wo_all
            best_macro_recall_wo_p = macro_recall
            best_macro_specificity_wo_p = macro_specificity
            best_unconfident_correct_count_wo_p = unconfident_correct_count
            best_unconfident_count_wo_p = unconfident_count
            best_ccp_threshold_wo_p = ccp_threshold
            best_precision_zero_div_count_wo_p = precision_zero_div_count
            best_recall_zero_div_count_wo_p = recall_zero_div_count
            best_specificity_zero_div_count_wo_p = specificity_zero_div_count
            best_macro_rac_ac_wo_p = macro_rac_ac
            best_micro_rac_ac_wo_p = micro_rac_ac
            best_average_reflection_count = average_reflection_count
            best_rac_ccp_wo_p = avg_rac_ccp
            best_urac_ccp_wo_p = avg_urac_ccp
            best_count1 = count1
            best_count2 = count2
            best_inc_error = inc_error
            best_cor_error = cor_error
            best_macro_incor_error_wo_p = macro_inc_error
            best_macro_cor_error_wo_p = macro_cor_error
            best_balanced_accuracy_wo_p = macro_balanced_accuracy

            # best_mean_error_wo_p = mean_error


    # import pdb; pdb.set_trace()
    ccp_values = []
    for i, row in df.iterrows():
        # import pdb; pdb.set_trace()
        if row['claim_uncertainty'] is None or row['claim_uncertainty'] == 0:
            continue
        for uncertainty in row['claim_uncertainty']:
            ccp_val = uncertainty.get('ccp')
            if ccp_val is not None:
                ccp_values.append(ccp_val)
            
    # import pdb; pdb.set_trace()
    assert ccp_threshold == args.ccp_threshold, "CCP threshold does not match the input argument."
    target = ccp_threshold
    quantile_v_for_train_data = find_quantile(target, ccp_values)


    if args.grid_search:
        quantile_v_for_best_ccp = find_quantile(best_ccp_threshold_wo_p, ccp_values)
        print(f"Best CCP Threshold (wo precision): {best_ccp_threshold_wo_p}")
        print(f'    CCP\'s corresponding Quantile Value: {quantile_v_for_best_ccp:.3f}')

        # print(f"    Best Macro Performance (wo precision): {best_macro_performance_wo_precision}")
        print(f"    Best Macro Precision (having all): {best_macro_precision_all_wo_p}")
        print(f"    Best Macro Precision (wo unconfident): {best_macro_precision_wo_unconfident_wo_p}")
        print(f"    Best Macro Precision (excluding all): {best_macro_precision_wo_all_wo_p}")
        print(f"    Best Macro Recall: {best_macro_recall_wo_p}")
        print(f"    Best Macro Specificity: {best_macro_specificity_wo_p}")
        print(f"    Uncertain Correct: {best_unconfident_correct_count_wo_p}/{best_unconfident_count_wo_p}")
        print(f"    Precision Zero Division Count: {best_precision_zero_div_count_wo_p}")
        print(f"    Recall Zero Division Count: {best_recall_zero_div_count_wo_p}")
        print(f"    Specificity Zero Division Count: {best_specificity_zero_div_count_wo_p}")
        print(f'    Best Macro (S+P)/2: {best_balanced_accuracy_wo_p}')
        print(f"    Best Macro RAC/AC: {best_macro_rac_ac_wo_p}")
        print(f"    Best Micro RAC/AC: {best_micro_rac_ac_wo_p}")
        print(f"    Best Average Reflection Count: {best_average_reflection_count}")
        print(f"    Best RAC CCP: {best_rac_ccp_wo_p}")
        print(f"    Best URAC CCP: {best_urac_ccp_wo_p}")
        print(f"    Best Unfound RAC CCP: {best_count1}")
        print(f"    Best Unfound URAC CCP: {best_count2}")
        print(f"    Best incorrect's ccp deviation: {best_inc_error}")
        print(f"    Best correct's ccp deviation: {best_cor_error}")
        print(f"    Best Macro Incorrect's ccp deviation: {best_macro_incor_error_wo_p}")
        print(f"    Best Macro Correct's ccp deviation: {best_macro_cor_error_wo_p}")


        # print(f"    Best Mean Error: {best_mean_error_wo_p}")

    print(f'Training Data\'s CCP Threshold (wo precision): {ccp_threshold}')
    print(f"    CCP's corresponding Quantile Value: {quantile_v_for_train_data:.3f}")
    # print(f'    Last Macro Performance (wo precision): {macro_performace_wo_precision}')
    print(f'    Last Macro Precision (having all): {macro_precision_all}')
    print(f'    Last Macro Precision (wo unconfident): {macro_precision_wo_unconfident}')
    print(f'    Last Macro Precision (excluding all): {macro_precision_wo_all}')
    print(f'    Last Macro Recall: {macro_recall}')
    print(f'    Last Macro Specificity: {macro_specificity}')
    print(f'    Last Uncertain Correct: {unconfident_correct_count}/{unconfident_count}')
    print(f'    Last Precision Zero Division Count: {precision_zero_div_count}')
    print(f'    Last Recall Zero Division Count: {recall_zero_div_count}')
    print(f'    Last Specificity Zero Division Count: {specificity_zero_div_count}')
    print(f'    Last (R+S)/2: {macro_balanced_accuracy}')
    print(f'    Last Macro RAC/AC: {macro_rac_ac}')
    print(f'    Last Micro RAC/AC: {micro_rac_ac}')
    print(f'    Last Average Reflection Count: {average_reflection_count}')
    print(f'    Last RAC CCP: {avg_rac_ccp}')
    print(f'    Last URAC CCP: {avg_urac_ccp}')
    print(f'    Last Unfound RAC CCP: {count1}')
    print(f'    Last Unfound URAC CCP: {count2}')
    print(f'    Last incorrect\'s ccp deviation: {inc_error}')
    print(f'    Last correct\'s ccp deviation: {cor_error}')
    print(f'    Last Macro Incorrect\'s ccp deviation: {macro_inc_error}')
    print(f'    Last Macro Correct\'s ccp deviation: {macro_cor_error}')
    print(f'    Last ccp diff: {avg_urac_ccp-avg_rac_ccp}')

    # save the final result
    df.to_json(args.output_file, index=False, orient='records', lines=True)
    

if __name__ == '__main__':
    main()
