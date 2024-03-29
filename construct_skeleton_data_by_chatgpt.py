import os
import openai
import numpy as np
import backoff
import argparse

openai.api_key="xxxxx"  #### "xxxxx" is openai api_key


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.Timeout, openai.error.APIError, openai.error.ServiceUnavailableError))
def generate_response_by_chat(prompt):
  chat_completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo-0613", 
  messages=[{"role": "system", "content": "You are a powerful syntax analyzer."},
            {"role": "user", "content": prompt}], 
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
def generate_prompt(args):
  questions = []
  prompt_before = "Please generate the question word phrase and the auxiliary verb in the sentence (if any).\nQuestion: "
  with open(os.path.join(args.fileName, args.questionName)) as f:
    for line in f.readlines():
      line = line.strip().lower()
      questions.append(line)
  prompts = [prompt_before + q for q in questions]
  return prompts

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--fileName', default = 'dataset/WQ', required=True)
  parser.add_argument('--questionName', default = 'train_question_gold.txt', required=True)
  parser.add_argument('--skeletonName', default = 'train_skeleton_chatgpt.txt', required=True)
  args = parser.parse_args() 
  
  #### generate the prompt  ###
  prompts = generate_prompt(args)
  response = []
  print("Starting!!!!")

  ########## generate skeleton by ChatGPT  ##########
  for idx, prompt in enumerate(prompts):
    with open(os.path.join(args.fileName, args.skeletonName),"a") as f:
      temp_response = generate_response_by_chat(prompt)
      response += temp_response
      f.write(temp_response[0] + "\n")

  
  print("Ending!!!!")
  
