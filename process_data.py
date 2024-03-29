
import os
import torch
import torch.nn as nn
import json
import pickle
import argparse
import numpy as np
from nltk import word_tokenize
from tqdm import tqdm
import re
import sys
import random
from transformers import BartTokenizer


def getNovelSubgraph(subkg, answer):
    g_nodes = subkg['g_node_names']
    g_edges = subkg['g_edge_types']
    g_adj = subkg['g_adj']
    g_edges = list(g_edges.values())
    relations = []
    seq = []
    for key, value in g_adj.items():
        subject = g_nodes[key]
        if subject == "none":
            continue
        else:
            for k in list(value.keys()):
                obj = g_nodes[k]
                if obj == "none" and k in list(g_adj.keys()):
                    value.update(g_adj[k])
    for key, value in g_adj.items():
        subject = g_nodes[key]
        if subject == "none":
            continue
        else:
            for k, relation in value.items():
                obj = g_nodes[k]
                if obj == "none":
                    continue
                else:
                    subject = subject.strip().lower()
                    obj = obj.strip().lower()
                    relation = relation.strip().lower()
                    relations.append(relation)
                    relation = relation.strip().split('/')[-1]
                    if relation.find('_')!=-1:
                        relation = relation.split('_')
                        relation = ' '.join(relation).strip()
                    fact = "<{}, {}, {}>".format(subject, relation, obj)
                    seq.append(fact)
    subkg = ", ".join(seq)                      
    return subkg, relations

def get_subkg_seq(subkg):
    seq = []
    maskList = []
    g_nodes = subkg['g_node_names']
    g_edges = subkg['g_edge_types']
    g_adj = subkg['g_adj']
    all_subjects = []
    all_objects = []
    relations = []
    for key, value in g_adj.items():
        subject = g_nodes[key]
        all_subjects.append(subject)
        for k, relation in value.items():
            obj = g_nodes[k]
            all_objects.append(obj)
            #####PQ relation is a list 
            # relation = relation[0]
            # relations.append(relation)
            # if relation.find('/') >= 0:
            #     relation = relation.strip().split('/')[-1]          

            #WQ relation is a str
            relation = relation.strip().split('/')[-1]

            if relation.find('_')!=-1:
                relation = relation.split('_')
                relation = ' '.join(relation).strip()
            fact = "<{}, {}, {}>".format(subject, relation, obj)
            seq.append(fact)
    subkg = ", ".join(seq)  
    return subkg, relations 


def encode_soft_skeleton_dataset(args, dataset, tokenizer, test, soft_tokens):
    max_seq_length = 496
    questions = []
    relations = []
    input_skeletons = []
    skeletons = []
    skeletons_train = []
    triple_nums = []
    answers = []
    subkgs = []
    answers_class = []
    count_answer = []
    for item in tqdm(dataset):
        question = item['outSeq']
        question = question.lower() 
        questions.append(question)
        subkg = item['inGraph']        
        answer = item['answers']   
        answer = [ans.lower() for ans in answer]  
       
        # subkg, relation = get_subkg_seq(subkg) #### PQ dataset
        subkg, node_names, relation, answer_class = getNovelSubgraph(subkg, answer) ####WQ dataset
        subkgs.append(subkg)

        if len(answer)>1:
            answer = [', '.join(answer)]
        if len(answer)==0:
            answer = ['']
        answers = answers + answer     
      
        relations.append(relation)

    s = [i +' | ' + j for i, j in zip(subkgs, answers)] ### no relation

    with open(os.path.join(args.input_dir, 'train_skeleton.txt')) as f:
        for line in f.readlines():
            line = line.strip().lower()
            skeletons.append(line)
    
    input_ids = tokenizer.batch_encode_plus(s, max_length = max_seq_length, padding = "max_length", truncation = True, return_tensors="pt")
   
    input_ids['input_ids'] =  torch.cat([input_ids['input_ids'], torch.full((len(input_ids['input_ids']), soft_tokens), 50264)], 1)
    input_ids['attention_mask'] =  torch.cat([input_ids['attention_mask'], torch.full((len(input_ids['attention_mask']), soft_tokens), 1)], 1)

    source_ids = np.array(input_ids['input_ids'], dtype = np.int32)
    source_mask = np.array(input_ids['attention_mask'], dtype = np.int32)
   

    if not test:
        target_ids = tokenizer.batch_encode_plus(skeletons, max_length = max_seq_length, padding = "max_length", truncation = True, return_tensors="pt")
        target_ids = np.array(target_ids['input_ids'], dtype = np.int32)
    else:
        target_ids = np.array([], dtype = np.int32)

    return source_ids, source_mask, target_ids   

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_name_or_path', required=True)
    args = parser.parse_args()
    print('Loading!!!!')
    test_set = []
    train_set = []
    val_set = []
    
    with open(os.path.join(args.input_dir, 'test.json')) as f:
        for line in f.readlines():
            line = line.strip()
            line = json.loads(line)
            test_set.append(line)
     
    with open(os.path.join(args.input_dir, 'train.json')) as f:
        for line in f.readlines(): 
            line = json.loads(line.strip())
            train_set.append(line)
    with open(os.path.join(args.input_dir, 'dev.json')) as f:
        for line in f.readlines():
            line = json.loads(line.strip())
            val_set.append(line)
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
   
    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)
    for name, dataset in zip(('train', 'val', 'test'), (train_set, val_set, test_set)):
        print('Encode {} set'.format(name))
        outputs = encode_soft_skeleton_dataset(args, dataset, tokenizer,  name == 'test' or name == 'val', 16) 
        print(type(outputs))
        print('shape of input_ids of questions, attention_mask of questions, input_ids of sparqls, choices and answers:')
        with open(os.path.join(args.output_dir, '{}.pt'.format(name)), 'wb') as f:
            for o in outputs:
                print(o.shape)
                pickle.dump(o, f)
if __name__ == '__main__':    
    main()   
    
