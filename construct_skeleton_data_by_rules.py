import os
import numpy as np
import argparse

def build_skeleton_data_by_rules(args):   
    skeletons = []
    with open(os.path.join(args.fileName, args.questionName)) as f:
        arr = ['what', 'which', 'where', 'who', 'whom', 'whose', 'how', 'how much', 'how many', 'how old', 'when', 'why']
        vb = ['in', 'on', 'to', 'by', 'for', 'of', 'from', 'with', 'about', 'at', 'during', 'besides', 'near', 'behind', 'across', 'into', 'under', 'through']
        vb_help = ['is', 'was', 'are', 'were', 'am','do', 'did', 'does']
        for idx, line in enumerate(f.readlines()):
            line = line.strip().lower().split(" ")
            inters = [v for v in line if v in arr]    
            inters_vb_help = [v for v in line if v in vb_help]           
            if len(inters) != 0:
                ix_list = [i for i, v in enumerate(line) if v == inters[-1]] 
                if (len(inters_vb_help)!=0):
                    idx_list = [i for i, v in enumerate(line) if v == inters_vb_help[0]]
                    if(line[0] in arr):
                        skeletons.append(" ".join(line[0: idx_list[0]+1]) +  " _ ?")
                    elif (line[1] in arr):
                        if (line[0] in vb):
                            skeletons.append(" ".join(line[0: idx_list[0]+1]) +  " _ ?")
                        else:
                            skeletons.append("_" + " " + line[1] + " _ ?")
                            
                    elif(inters[-1] in arr and line[ix_list[-1]-1] in vb):  
                        if(inters[-1] == line[-1]):
                            skeletons.append("_ " + line[ix_list[-1]-1] + " " + inters[-1] + " ?")
                        else:
                            if ix_list[-1] < idx_list[0]:
                                skeletons.append("_ " + " ".join(line[ix_list[-1]-1 : idx_list[0]+1]) + " _ ?") 
                            else:
                                skeletons.append("_ " + line[ix_list[-1]-1] + " " + inters[-1] + " _ ?")  
                        
                    elif(inters[-1] in arr and line[ix_list[-1]-1] not in vb):
                        if(inters[-1] == line[-1]):
                            skeletons.append("_ " + inters[-1] + " ?")
                        else:
                            if ix_list[-1] < idx_list[0]:
                                skeletons.append("_ " + " ".join(line[ix_list[-1] : idx_list[0]+1]) + " _ ?")
                            else:
                                skeletons.append("_ " + inters[-1] + " _ ?")  
                else:
                    if(line[0] in arr):
                        skeletons.append(line[0] +  " _ ?")
                    elif (line[1] in arr):
                        if (line[0] in vb):
                            skeletons.append(line[0] + " " + line[1] + " _ ?")
                        else:
                            skeletons.append("_" + " " + line[1] + " _ ?")
                            
                    elif(inters[-1] in arr and line[ix_list[-1]-1] in vb):  
                        if(inters[-1] == line[-1]):
                            skeletons.append("_ " + line[ix_list[-1]-1] + " " + inters[-1] + " ?")
                        else:
                            skeletons.append("_ " + line[ix_list[-1]-1] + " " + inters[-1] + " _ ?")  
                        
                    elif(inters[-1] in arr and line[ix_list[-1]-1] not in vb):
                        if(inters[-1] == line[-1]):
                            skeletons.append("_ " + inters[-1] + " ?")
                        else:
                            skeletons.append("_ " + inters[-1] + " _ ?") 
            else:
                skeletons.append("_ ?")  
        np.savetxt(os.path.join(args.fileName, args.skeletonName), skeletons, fmt='%s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fileName', default = 'dataset/WQ', required=True)
    parser.add_argument('--questionName', default = 'train_question_gold.txt', required=True)
    parser.add_argument('--skeletonName', default = 'train_skeleton_rules.txt', required=True)
    args = parser.parse_args() 
    build_skeleton_data_by_rules(args)

