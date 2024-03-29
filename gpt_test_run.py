import os
import openai
import numpy as np
import backoff
import tiktoken
import sys
import argparse

openai.api_key="xxxxx"  #### "xxxxx" is openai api_key

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.Timeout, openai.error.APIError, openai.error.ServiceUnavailableError))
def generate_response(prompt):
	response = openai.Completion.create(
	  model="text-davinci-003",  
	  prompt = prompt,
	  temperature = 0.7, 
	  max_tokens = 100,  
          n = 10,
	  top_p = 1,
	  frequency_penalty = 0,
	  presence_penalty = 0
	)
	final_response = [response.choices[i]["text"].strip() for i in range(len(response.choices))]
	return final_response

#### count tokens###
def count_tokens(text):
  encoding = tiktoken.get_encoding("cl100k_base")
  token_count = len(encoding.encode(text))
  return token_count

#### construct the prompt ####
def generate_prompt(testSubkgFile, testAnswerFile, testSkeletonFile, trainSubkgFile, trainAnswerFile, trainSkeletonFile, trainQuestionFile, fewShotIndexFile):
  prompt_before = "Please generate a detailed and specific complex question using the provided skeleton and the information in the triples related to the answer. The question should include all relevant details from the triples while avoiding directly mentioning the answer in the question itself." 
  prompt_after1 = "\nTriples: "
  prompt_after2 = "\nAnswer: "
  prompt_after3 = "\nSkeleton: "
  prompt_after4 = "\nQuestion: "
  testSkeletons = []
  trainSkeletons = []
  trainQuestions = []
  with open(testSkeletonFile) as f:
    for line in f.readlines():
      line = line.strip().lower()
      testSkeletons.append(line)
  with open(trainSkeletonFile) as f:
    for line in f.readlines():
      line = line.strip().lower()
      trainSkeletons.append(line)
  with open(trainQuestionFile) as f:
    for line in f.readlines():
      line = line.strip().lower()
      trainQuestions.append(line)
  testAnswers = np.load(testAnswerFile)
  testSubkgs = np.load(testSubkgFile)
  trainAnswers = np.load(trainAnswerFile)
  trainSubkgs = np.load(trainSubkgFile)
  fewShotIndex = np.load(fewShotIndexFile)  
  prompts = []
  trainQuestions = np.array(trainQuestions)
  trainSkeletons = np.array(trainSkeletons)
  for fewIndex, testKg, testAns, testSke in zip(fewShotIndex, testSubkgs, testAnswers, testSkeletons):    
    temp_train_subkgs = trainSubkgs[fewIndex]
    temp_train_answers = trainAnswers[fewIndex]
    temp_train_skeletons = trainSkeletons[fewIndex]
    temp_train_questions = trainQuestions[fewIndex]
    temp_prompt = []    
    for i, j, m, n in zip(temp_train_subkgs, temp_train_answers, temp_train_skeletons, temp_train_questions):
      s = prompt_after1 + i + prompt_after2 + j[0] + prompt_after3 + m + prompt_after4 + n
      temp_prompt.append(s)
  
    temp_prompt.reverse()
    prompt_x = "".join(temp_prompt)
    prompt = prompt_before + prompt_x + prompt_after1 + testKg + prompt_after2 + testAns[0] + prompt_after3 + testSke + prompt_after4
    while count_tokens(prompt) > 3500: ### To avoid exceeding the maximum number of tokens, we set 3500
      temp_prompt = temp_prompt[:-1]
      prompt_x = "".join(temp_prompt)
      prompt = prompt_before + prompt_x + prompt_after1 + testKg + prompt_after2 + testAns[0] + prompt_after3 + testSke + prompt_after4

    prompts.append(prompt)
  
  return prompts


if __name__ == '__main__':
 
  #########WQ Dataset#############
  testSubkgFile = "./dataset/WQ/test_subkgs.npy"
  testAnswerFile = "./dataset/WQ/test_answers.npy"
  testSkeletonFile = "./dataset/WQ/predict_test_skeleton.txt"
  trainSubkgFile =  "./dataset/WQ/train_subkgs.npy"
  trainAnswerFile = "./dataset/WQ/train_answers.npy"
  trainSkeletonFile = "./dataset/WQ/train_skeleton.txt"
  trainQuestionFile = "./dataset/WQ/train_question_gold.txt"
  fewShotIndexFile = "./dataset/WQ/similarity_by_embeddings_and_skeleton_top_16_idx.npy"
  
  prompts = generate_prompt(testSubkgFile, testAnswerFile, testSkeletonFile, trainSubkgFile, trainAnswerFile, trainSkeletonFile, trainQuestionFile, fewShotIndexFile)
  response = []
  print("Starting!!!!")
  for prompt in prompts:   
    temp_response = generate_response(prompt) ####text-davinci-003
    response.append(temp_response)
  np.save("output/test_output_davinci_few_16_top_10.npy", response)
  
  print("Ending!!!!")
  
