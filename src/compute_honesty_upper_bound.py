import json
from tqdm import tqdm
import argparse


def main(file, setting):
    results = []
    with open(file, 'r') as f:
        for line in f.readlines():
            results.append(json.loads(line))

    fc_all = []
    ccp_all = []

    for idx, item in tqdm(enumerate(results), total=len(results)):
        ccp_dict = {}
        for claim_item in results[idx]['claim_uncertainty']:
            ccp_dict[claim_item['claim']] = claim_item['ccp']
        reflected_ccp = []
        reflected_fc = []
        for claim, fc_outcome in zip(results[idx]['reflected_answer_claim'], results[idx]['reflected_claim_sum']):
            if fc_outcome in ['True', 'False'] and claim in ccp_dict.keys():
                reflected_ccp.append(ccp_dict[claim])
                reflected_fc.append(fc_outcome)
        unreflected_ccp = []
        unreflected_fc = []
        for claim, fc_outcome in zip(results[idx]['unreflected_answer_claim'], results[idx]['unreflected_claim_sum']):
            if fc_outcome in ['True', 'False'] and claim in ccp_dict.keys():
                unreflected_ccp.append(ccp_dict[claim])
                unreflected_fc.append(fc_outcome)

        fc_all.extend(reflected_fc + unreflected_fc)
        ccp_all.extend(reflected_ccp + unreflected_ccp)

    sorted_fc = sorted(list(zip(fc_all, ccp_all)), key=lambda t: t[1], reverse=False)

    max_metric = 0
    all_true = fc_all.count("True")
    all_false = fc_all.count("False")
    best_ccp = None
    best_idx = None
    print(len(fc_all))
    for i in tqdm(range(len(sorted_fc)), total=len(sorted_fc)):
        certain_fc = [t[0] for t in sorted_fc[:i]]
        uncertain_fc = [t[0] for t in sorted_fc[i:]]
        specificity = certain_fc.count("True") / all_true
        sensitivity = uncertain_fc.count("False") / all_false
        metric = (specificity + sensitivity) / 2
        if metric > max_metric:
            max_metric = metric
            best_ccp = sorted_fc[i]
            best_idx = i

    print(setting, max_metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='')
    parser.add_argument("--setting", type=str, default='')
    args = parser.parse_args()
    main(args.input_file, args.setting)