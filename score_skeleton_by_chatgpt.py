import os
import openai
import numpy as np
import backoff
import argparse


openai.api_key="xxxxx"  #### "xxxxx" is openai api_key


##### chatgpt ####
@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.Timeout, openai.error.APIError, openai.error.ServiceUnavailableError))
def generate_response_by_chat(prompt):
  chat_completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo-0613", 
  messages = [{"role": "system", "content": "You are a scoring system used to evaluate the alignment of the skeleton to the question, where the skeleton is extracted from the question and just contains the question word and the auxiliary verb (if any)."},
              {"role": "user", "content" : prompt}], 
  n=1, 
  temperature=0,
  max_tokens=256,
  top_p=1, 
  frequency_penalty=0,
  presence_penalty=0 
  )
  response = [chat_completion.choices[i].message["content"].strip() for i in range(len(chat_completion.choices))]
  return response

#### construct the prompt ####
def generate_score_prompt(filePath, fileQ, fileS1, fileS2, notIndex):
  questions = []
  skeletons1 = []
  skeletons2 = []
  prompt_before = "Please score the following two skeletons according to the alignment of the skeleton to the question. Scores range from 0 to 1, where a higer score indicates higher accuracy and completeness of the skeleton. Please fairly give the scores for skeleton 1 and skeleton 2, do not output other content. \nQuestion: "
  prompt_after1 = "\n Skeleton 1: "
  prompt_after2 = "\n Skeleton 2: "
  with open(os.path.join(args.fileName, args.questionName)) as f:
    for line in f.readlines():
      line = line.strip().lower()
      questions.append(line)
  with open(os.path.join(args.fileName, args.skeletonName1)) as f:
    for line in f.readlines():
      line = line.strip().lower()
      skeletons1.append(line)
  with open(os.path.join(args.fileName, args.skeletonName2)) as f:
    for line in f.readlines():
      line = line.strip().lower()
      skeletons2.append(line)
  questions = np.array(questions)[notIndex]
  skeletons1 = np.array(skeletons1)[notIndex]
  skeletons2 = np.array(skeletons2)[notIndex]
  prompts = [prompt_before + q + prompt_after1 + s1 + prompt_after2 + s2 for q, s1, s2 in zip(questions, skeletons1, skeletons2)]
  return prompts 

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fileName', default = 'dataset/WQ', required=True)
  parser.add_argument('--questionName', default = 'train_question_gold.txt', required=True)
  parser.add_argument('--skeletonName1', default = 'train_skeleton_rules.txt', required=True)
  parser.add_argument('--skeletonName2', default = 'train_skeleton_chatgpt.txt', required=True)
  parser.add_argument('--skeletonScore', default = 'train_skeleton_score_by_chatgpt.txt', required=True)
  args = parser.parse_args() 

  #### generate the prompt ####
  prompts =  generate_score_prompt(args)
  response = []
  print("Starting!!!!")

 ############## score skeleton by ChatGPT ###################
  for idx, prompt in enumerate(prompts):
    temp_response = generate_response_by_chat(prompt)
    response += temp_response
    f.write(temp_response[0] + "\n")
  np.save(os.path.join(args.fileName, args.skeletonScore), response)
  
  print("Ending!!!!")
  
