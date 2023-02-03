# DietCode

This repo provides the code for reproducing the experiments in DietCode ("Diet Code is Healthy: Simplifying Programs for Pre-Trained Models of Code") published at FSE 2022. DietCode is a program simplification method which aims for light-weighted leverage of pre-trained programming language models.

## Requirements

* [Python3](https://www.linuxbabe.com/ubuntu/install-python-3-6-ubuntu-16-04-16-10-17-04) 
* [PyTorch](https://pytorch.org/)

## Quick Start

### Prepare the dataset

As this project conducted a empirical study on [CodeBERT](https://arxiv.org/pdf/2002.08155.pdf), we just use the same dataset as CodeBERT.

The raw data can be downloaded from [CodeSearchNet](https://github.com/github/CodeSearchNet) and the preprocessed dataset can be downloaded by following the step on [CodeBERT](https://github.com/microsoft/CodeBERT/tree/master/CodeBERT).

### Obtain the attention score of statements

You have to fill the `java_map` or `python_map` in each `prune.py` file in the downstream task folders.

The following code provides an example which contains the processed data of java statement attention and python statement attention respectively.
```python
java_statement_classification_map = {
    'try': 0.0029647741585358297,
    'catch': 0.0025092298911411127,
    'finally': 0.003843427080920313,
    'break': 0.002504047667805474,
    'continue': 0.0025862206572769947,
    'return': 0.003415540177420338,
    'throw': 0.002409465431368352,
    'annotation': 0.0028472381356659383,
    'while': 0.002679541985162062,
    'for': 0.002537917195113055,
    'if': 0.0025404393423889915,
    'switch': 0.0025462886222332426,
    'expression': 0.0032153782437548553,
    'synchronized': 0.0023616513586135323,
    'case': 0.002325461992369871,
    'method': 0.004119854399696806,
    'variable': 0.0024165516139185456,
    'logger': 0.002416770362746685,
    'setter': 0.0026460245897558608,
    'getter': 0.0025480630285627617,
    'function': 0.0027256693629142824,
}
```

```python
python_statement_classification_map = {
    'try': 0.001126432983856304,
    'break': 0.000587459733777285,
    'finally': 0.001424176323798515,
    'continue': 0.000525201820979588,
    'return': 0.003557705224077524,
    'annotation': 0.001515482513624964,
    'while': 0.001181877585921989,
    'for': 0.001555825490959887,
    'if': 0.002425228787754846,
    'expression': 0.001606519048049276,
    'method': 0.006854916776239256,
    'variable': 0.001436622015556903,
    'function': 0.001785857572541693,
}
```

and get the low rated attention tokens in a low_rated_tokens file which will be used to have advanced pruning after pruning the statements.


### Reproduce the experimental results

#### Code Search

If you want to collect the attention of the tokens and statments you can add `--output_attention` when training the codesearch downstream task and the ./utils/analyse.py can help you analyse the attention to generate low_rated_tokens and statment attentions.

You can change the `model_type` argument from roberta to codet5 if you want to test DietCodeT5.

Also you can modify `prune_strategy` to random or token to test random strategy or token_frequency strategy.

Training:

```
lang=java
pretrained_model=microsoft/codebert-base

python3 run_classifier.py \
	--model_type roberta \
	--task_name codesearch \
	--do_train \
	--do_eval \
	--eval_all_checkpoints \
	--train_file train.txt \
	--dev_file valid.txt \
	--max_seq_length 120 \
	--per_gpu_train_batch_size 64 \
	--per_gpu_eval_batch_size 64 \
	--learning_rate 1e-5 \
	--num_train_epochs 4 \
	--output_attention \
	--lang java \
	--gradient_accumulation_steps 1 \
	--overwrite_output_dir \
	--prune_strategy slim \
	--data_dir ../data/codesearch/train_valid/$(lang) \
	--output_dir ./models/$(lang)/output \
	--tokenizer_name microsoft/codebert-base \
	--model_name_or_path $(pretrained_model)
```

Evaluating:

```
lang=java
pretrained_model=microsoft/codebert-base
idx=0

python3 run_classifier.py \
	--model_type roberta \
	--model_name_or_path microsoft/codebert-base \
	--task_name codesearch \
	--do_predict \
	--output_attention \
	--prune_strategy slim \
	--output_dir ./models/$(lang)/slim_origin_5 \
	--data_dir ../data/codesearch/test/$(lang) \
	--max_seq_length 120 \
	--per_gpu_train_batch_size 32 \
	--per_gpu_eval_batch_size 32 \
	--learning_rate 1e-5 \
	--num_train_epochs 8 \
	--test_file batch_$(idx).txt \
	--pred_model_dir ./models/$(lang)/slim/checkpoint-best \
	--tokenizer_name microsoft/codebert-base \
	--test_result_dir ./results/$(lang)/slim_tiny/$(idx)_batch_result.txt
```

and running `python3 mrr.py` to get the mrr score.

#### Code Summarization

You can change the `model_type` argument from RoBERTa to CodeT5 if you want to fine-tune or test from CodeT5 model.

Training:

```
lang=java
lr=5e-5
batch_size=24
beam_size=10

target_length=128
data_dir=../data/code2nl/CodeSearchNet

train_file=$(data_dir)/$(lang)/train.jsonl
dev_file=$(data_dir)/$(lang)/valid.jsonl
eval_steps=10
train_steps=30000
pretrained_model=microsoft/codebert-base
output_dir=model/$(lang)/slim-tiny

python3 run.py --do_train --do_eval --model_type roberta \
        --tokenizer_name microsoft/codebert-base \
        --model_name_or_path $(pretrained_model) --train_filename $(train_file) \
        --prune_strategy slim \
        --dev_filename $(dev_file) --output_dir $(output_dir) --max_source_length 100 \
        --max_target_length $(target_length) --beam_size $(beam_size) --train_batch_size $(batch_size) \
        --eval_batch_size $(batch_size) --learning_rate $(lr) --train_steps $(train_steps) --eval_steps $(eval_steps)
```

Evaluating:

```
lang=java
beam_size=10
batch_size=128
target_length=128
output_dir=model/$(lang)/slim-tiny
data_dir=../data/code2nl/CodeSearchNet
dev_file=$(data_dir)/$(lang)/valid.jsonl
test_file=$(data_dir)/$(lang)/test.jsonl
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin

python3 run.py \
	--do_test \
	--model_type roberta \
	--tokenizer_name microsoft/codebert-base \
	--prune_strategy slim \
	--model_name_or_path microsoft/codebert-base \
	--dev_filename $(dev_file) \
	--test_filename $(test_file) \
	--output_dir $(output_dir) \
	--max_source_length 100 \
	--max_target_length $(target_length) \
	--beam_size $(beam_size) \
	--load_model $(output_dir)/checkpoint-best-bleu/pytorch_model.bin
	--eval_batch_size $(batch_size)
```

