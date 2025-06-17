import json
import argparse


def main(file, setting):

    results = []
    with open(file, 'r') as f:
        for line in f.readlines():
            results.append(json.loads(line))


    reflected_all = []
    unreflected_all = []
    unreflected_true_portion = []
    sensitivity = []
    specificity = []

    for idx, item in enumerate(results):

        reflected = [i for i in results[idx]['reflected_claim_sum'] if i in ['True', 'False']]
        unreflected = [i for i in results[idx]['unreflected_claim_sum'] if i in ['True', 'False']]
        if (len(reflected) + len(unreflected)) > 0:
            unreflected_true_portion.append(unreflected.count("True") / (len(reflected) + len(unreflected)))
        reflected_all.extend(reflected)
        unreflected_all.extend(unreflected)
        if reflected.count("True") + unreflected.count("True") > 0:
            specificity.append(unreflected.count("True") / (reflected.count("True") + unreflected.count("True")))
        if reflected.count("False") + unreflected.count("False") > 0:
            sensitivity.append(reflected.count("False") / (reflected.count("False") + unreflected.count("False")))

    print(len(reflected_all) / (len(reflected_all) + len(unreflected_all)))
    print(f"===={setting}====")
    # print("Count (True - False)", ((unreflected_all.count("True") + reflected_all.count("True")) - (unreflected_all.count("False") + reflected_all.count("False"))) / (len(reflected_all) + len(unreflected_all)))
    print("Overall Acc.", (reflected_all.count("True") + unreflected_all.count("True")) / (len(unreflected_all) + len(reflected_all)))
    if len(reflected_all) > 0:
        print("Model unsure Acc.", reflected_all.count("True") / len(reflected_all))
    print("Model sure Acc.", unreflected_all.count("True") / len(unreflected_all))

    print("Macro Specificity", sum(specificity) / len(specificity))
    print("Macro Sensitivity", sum(sensitivity) / len(sensitivity))
    print("Macro Avg.", ((sum(specificity) / len(specificity)) + (sum(sensitivity) / len(sensitivity))) / 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default='')
    parser.add_argument("--setting", type=str, default='')
    args = parser.parse_args()
    main(args.input_file, args.setting)
