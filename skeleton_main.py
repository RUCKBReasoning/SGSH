import os
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import shutil
import json
from tqdm import tqdm
from datetime import date
from utils.misc import MetricLogger, seed_everything, ProgressBar
from data import DataLoader
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
from torch.optim import AdamW
import logging
import time
import random
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import nltk
import numpy as np
import warnings
from soft_embedding import SoftEmbedding

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info("Create train_loader and val_loader.........")
    train_pt = os.path.join(args.output_dir, 'train.pt')
    train_loader = DataLoader(train_pt, args.batch_size, training=True)
    logging.info("Create model.........")

    model_class, tokenizer_class = (BartForConditionalGeneration, BartTokenizer)    
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    n_tokens = 16
    initialize_from_vocab = True
    s_wte = SoftEmbedding(model.get_input_embeddings(), n_tokens=n_tokens, initialize_from_vocab=initialize_from_vocab)
    model.set_input_embeddings(s_wte)
    model = torch.nn.DataParallel(model, device_ids = [0])
    model = model.to(device)   
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)   
    start = time.time()
    # Train!
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(train_loader.dataset))
    logging.info("  Num Epochs = %d", args.num_train_epochs)   
    global_step = 0
    tr_loss = 0.0
    optimizer.zero_grad()
    bleu1MaxScore = 0.0
    bleu2MaxScore = 0.0
    bleu3MaxScore = 0.0
    bleu4MaxScore = 0.0
    semanticMaxScore = 0.0
    for i in range(int(args.num_train_epochs)):
        model.train()
        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(device) for t in batch)
            pad_token_id = tokenizer.pad_token_id
            source_ids, source_mask, y = batch[0], batch[1], batch[2]
            
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone()
            
            lm_labels[y[:, 1:] == pad_token_id] = -100
            outputs = model(input_ids = source_ids.to(device), attention_mask = source_mask.to(device), decoder_input_ids = y_ids.to(device), labels = lm_labels.to(device))
            loss = outputs[0]        
            loss.mean().backward()
            print("loss:",loss.mean().item())
            tr_loss += loss.mean().item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        if 'cuda' in str(device):
            torch.cuda.empty_cache()
        bleu1_score, bleu2_score, bleu3_score, bleu4_score, rouge_score = test(model, tokenizer, args, i)

        if i == 0 or bleu1MaxScore < bleu1_score:
            bleu1MaxScore = bleu1_score
            torch.save(model.module.state_dict(), os.path.join(args.output_dir, 'param_bart_best.pt'), _use_new_zipfile_serialization = False)             
    end = time.time()
    t = (end-start)/3600
    print("run time: {} h".format(t))
    return model, tokenizer

def test(model, tokenizer, args, i):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.module.to(device)
    model.eval()
    with torch.no_grad():
        all_outputs = []
        test_pt = os.path.join(args.output_dir, 'val.pt')
        test_loader = DataLoader(test_pt, args.batch_size)
        for batch in tqdm(test_loader, total=len(test_loader)):            
            batch = batch[:2]
            source_ids, source_mask= [x.to(device) for x in batch]
            outputs = model.generate(
                input_ids = source_ids,
                num_beams = 5,
                max_length = 128,
                use_cache=False
            )
           
            all_outputs.extend(outputs.cpu().numpy())
    outputs = [tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in all_outputs]
    temp_outputs = []
    for line in outputs:
        line = line.strip().lower()
        l = line.split(" ")
        ll = l[-1].split("?")
        if ll[0]!= "":
            l[-1] = ll[0]
            l.append("?")
            l = " ".join(l)
            l = " " if len(l.strip()) == 0 else l.strip()
            temp_outputs.append(l)
        else:
            line = " " if len(line.strip()) == 0 else line.strip()
            temp_outputs.append(line)
    outputs = temp_outputs
    groundTruth = []

    with open("./dataset/WQ/dev_skeleton.txt") as f:
        for line in f.readlines():
            groundTruth.append(line.strip().lower())
    
    #calculate bleu score
 
    bleu1_list =[]
    bleu2_list =[]
    bleu3_list =[]
    bleu4_list =[]
    
    rouge_score =[]
    rouge = Rouge()

    for pred, ground  in zip(outputs, groundTruth):      

        pred = nltk.word_tokenize(pred.strip())
        ground = nltk.word_tokenize(ground.strip())
        pred_new = ' '.join(pred)
        ground_new = ' '.join(ground)
        if len(pred)==0:
            pred_new = ' '
        rouge_l = rouge.get_scores(pred_new, ground_new)
        ro_score = rouge_l[0]['rouge-l']['r']
        ro_score = round(ro_score*100, 4)
        rouge_score.append(ro_score)
        
        
        bleu4 = sentence_bleu([ground],pred,weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1) #bleu4
        bleu4 = round(bleu4*100,4)
        bleu3 = sentence_bleu([ground],pred,weights=(0.33, 0.33, 0.33, 0), smoothing_function=SmoothingFunction().method1)
        bleu3 = round(bleu3*100,4)
        bleu2 = sentence_bleu([ground],pred,weights=(0.5, 0.5, 0, 0), smoothing_function=SmoothingFunction().method1)
        bleu2 = round(bleu2*100,4)
        bleu1 = sentence_bleu([ground],pred,weights=(1.0, 0, 0, 0), smoothing_function=SmoothingFunction().method1)
        bleu1 = round(bleu1*100,4)           
        
        bleu1_list.append(bleu1)
        bleu2_list.append(bleu2)
        bleu3_list.append(bleu3)
        bleu4_list.append(bleu4)    
    
    return np.mean(bleu1_list), np.mean(bleu2_list), np.mean(bleu3_list), np.mean(bleu4_list), np.mean(rouge_score)

##### infer stage #######
def infer(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_class, tokenizer_class = (BartForConditionalGeneration, BartTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)

    ###### soft prompt #####
    n_tokens = 16
    initialize_from_vocab = True
    s_wte = SoftEmbedding(model.get_input_embeddings(), n_tokens=n_tokens, initialize_from_vocab=initialize_from_vocab)
    model.set_input_embeddings(s_wte)
    
    # load the best parameters
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'param_bart_best.pt')) )
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        all_outputs = []
        test_pt = os.path.join(args.output_dir, 'test.pt')
        test_loader = DataLoader(test_pt, args.batch_size)
        for batch in tqdm(test_loader, total=len(test_loader)):            
            batch = batch[:2]
            source_ids, source_mask= [x.to(device) for x in batch]
            outputs = model.generate(
                input_ids = source_ids,
                num_beams = 5,
                max_length = 128,
            )
            all_outputs.extend(outputs.cpu().numpy())
    outputs = [tokenizer.decode(output_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for output_id in all_outputs]

    #save generated skeletons
    np.savetxt(os.path.join(args.output_dir, 'predict_test_skeleton.txt'), outputs, fmt='%s')
    


def main():
    parser = argparse.ArgumentParser()
    #### input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_name_or_path', required = True)

    #### training parameters
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--learning_rate', default=5e-5, type = float)
    parser.add_argument('--num_train_epochs', default=20, type = int)
    parser.add_argument('--logging_steps', default=448, type = int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--isTrain', default = True)
    
    #### args display
    for k, v in vars(args).items():
        logging.info(k+':'+str(v))
    seed_everything(args.seed)

    #### training stage or inferring stage #####
    if args.isTrain:
        model, tokenizer = train(args)
    else:
        infer(args)
   
   
if __name__ == '__main__':
    main()


   
