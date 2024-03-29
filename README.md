SGSH
====
The Pytorch implementation of SGSH: Stimulate Large Language Models with Skeleton Heuristics for Knowledge Base Question Generation(NAACL 2024 Findings).

# Requirements
## 1. Environments
* Create a virtual environment by running the following command:
```
$ conda env create --name=SGSH --file=environment.yml
```
* Activate the environment using:
```
$ conda activate SGSH
```

## 2. Dataset
Our experiments contain two widely-used datasets, i.e., WebQuestions (WQ) and PathQuestions (PQ). The raw data of these datasets are from GitHub [Graph2Seq](https://github.com/hugochan/Graph2Seq-for-KGQG). You can directly use the datasets in our folder `dataset/`. 
* WQ: `dataset/` contains files for the WQ dataset.

* PQ: `dataset/` contains files for the PQ dataset.

More specifically, `WQ/` and `PQ/` mainly contain the following files:
* `train.json`, `dev.json`, and `test.json` are the data for train, dev, and test, respectively.

* `train_question_gold.txt`, `dev_question_gold.txt`, and `test_question_gold.txt` are the ground-truth questions for train data, dev data, and test data, respectively.
* `train_skeleton.txt` and `dev_skeleton.txt` are skeleton training data constructed using the automatic training data construction strategy we proposed.

# Quick Start for Running

## 1. Fine-tuning Skeleton Generator.
   
* Prepare the dataset for the skeleton generator by running the following command. Alternatively, You can directly use the built data in `dataset/WQ/train_skeleton.txt` and `dataset/WQ/dev_skeleton.txt` (Note: we take the WQ dataset as an example).

  * Extract skeletons using the rule-based method, execute:
  ```
  $ python construct_skeleton_data_by_rules.py --fileName './dataset/WQ' --questionName 'train_question_gold.txt' --skeletonName 'train_skeleton_rules.txt'
  ```
  * Generate skeletons using a ChatGPT-based skeleton generator, execute:
  ```
  $ python construct_skeleton_data_by_chatgpt.py --fileName './dataset/WQ' --questionName 'train_question_gold.txt' --skeletonName 'train_skeleton_chatgpt.txt'
  ```
  * Refine skeletons by ChatGPT-based skeleton quality evaluator, execute:
  ```
  $ python score_skeleton_by_chatgpt.py --fileName './dataset/WQ' --questionName 'train_question_gold.txt' --skeletonName1 'train_skeleton_rules.txt' --skeletonName2 'train_skeleton_chatgpt.txt' --skeletonScore 'train_skeleton_score_by_chatgpt.txt'
  ```
  * Prepare training data for training skeleton generator, execute:
   ```
   $ python process_data.py --input_dir './dataset/WQ' --output_dir './output' --model_name_or_path 'facebook/bart-base'
   ```
* To train the skeleton generator, execute:
```
$ python skeleton_main.py --input_dir './dataset/WQ' --output_dir './output' --model_name_or_path 'facebook/bart-base' --learning_rate 5e-5 --batch_size 16 --num_train_epochs 20
```
* To infer and acquire the generated skeleton on the test dataset (i.e., './dataset/WQ/predict_test_skeleton.txt'), execute:
```
$ python skeleton_main.py --isTrain False --input_dir './dataset/WQ' --output_dir 'output' --model_name_or_path 'facebook/bart-base' --batch_size 16 
```
## 2. To infer on GPT-3.5 (e.g., text-davinci-003) to obtain the generated questions, execute:
```
$ python gpt_test_run.py
```


