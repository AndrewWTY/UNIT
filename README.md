# Balancing Truthfulness and Informativeness with Uncertainty-Aware Instruction Fine-Tuning </a>
UNIT-Ref is a novel IFT paradigm to address hallucination by teaching LLMs to recognize their uncertainty and explicitly reflect it at the end of their responses. This is the official repository for [our paper](https://arxiv.org/abs/2502.11962).


## Setup
First, install python dependences
```console
python -m venv ccp
source ccp/bin/activate
pip install -r requirements_ccp.txt
deactivate
```
and 
```console
python -m venv vllm
source ccp/bin/activate
pip install -r requirements_vllm.txt
deactivate
```

## Prepare the Training Dataset
### To probe the CCP uncertainty given a dataset
For example to obtain Qwen/Qwen2.5-14B 's CCP uncertainty on an IFT dataset (e.g. lima.jsonl), download and put the lima.jsonl dataset into the folder data_to_probe_ccp and run:
```console
bash eval_pipeline.sh --model_name Qwen/Qwen2.5-14B --get_ccp_from_response lima.jsonl
```
You will find the result in evaluate_database/calibration_result/lima_calibrated_Qwen_Qwen2.5-14B.jsonl

### To classify the infomation-seeking tasks within a dataset
Then, to classify the information-seeking tasks within a dataset
```console
python instruction_classification.py \
    --instruction_file evaluate_database/calibration_result/lima_calibrated_Qwen_Qwen2.5-14B.jsonl \
    --output_file train_data/lima_calibrated_Qwen_Qwen2.5-14B_labelled.jsonl \
    --llm gpt-4o \
    --batch 20
```


### 1. UNIT-Reflection Dataset
To obtain the UNIT-Reflection Dataset, run:
```console
python make_data.py\
    --input_file evaluate_database/calibration_result/lima_calibrated_Qwen_Qwen2.5-14B_labelled.jsonl \
    --output_file train_data/lima_calibrated_Qwen_Qwen2.5-14B_labelled.jsonl \
```



## Training
We used [Alignment Handbook](https://github.com/huggingface/alignment-handbook/tree/main) to fine-tune our models using the dataset created by UNIT-Ref and UNIT-Cut, the sample training configs containing all our training hyperparameters used can be found in training_configs folder.


## Evaluation
To evaluate the fine-tuned checkpoints on Biography (bio) or WildHalu run:
```console
bash eval_pipeline.sh --model_name your_checkpoint_path --test_data [bio|wildhalu]
```

## Datasets
This repository contains 2 types of datasets:
- In datasets folder, we provide LIMA and LFRQA's dataset with their probed ccp values wrt. Llama3.1-8b and Qwen2.5-14B, uncertain claims and certain claims classified by quantile 75 is also included in these datasets.
- In src/inference_data, we provide WildHalu (500 subset we used) and Biography datasets for evaluation.
