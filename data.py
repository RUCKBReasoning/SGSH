import json
import pickle
import torch
from utils.misc import invert_dict

def collate(batch):
    batch = list(zip(*batch))
    source_ids = torch.stack(batch[0])
    source_mask = torch.stack(batch[1])
    if batch[-1][0] is None:
        target_ids = None
    else:
        target_ids = torch.stack(batch[2])
    return source_ids, source_mask, target_ids

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.source_ids, self.source_mask, self.target_ids = inputs
        self.is_test = len(self.target_ids)==0


    def __getitem__(self, index):
        source_ids = torch.LongTensor(self.source_ids[index])
        source_mask = torch.LongTensor(self.source_mask[index])
       
        if self.is_test:
            target_ids = None
           
        else:
            target_ids = torch.LongTensor(self.target_ids[index])
          
        return source_ids, source_mask, target_ids
    def __len__(self):
        return len(self.source_ids)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, question_pt, batch_size, training=False):
          
        inputs = []
        with open(question_pt, 'rb') as f:
            for _ in range(3):
                inputs.append(pickle.load(f))
        dataset = Dataset(inputs)
        self.len = len(dataset)
        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate, 
            )
     
