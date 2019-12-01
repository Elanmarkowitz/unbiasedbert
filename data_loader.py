#torch.nn.utils.rnn.pad_sequence([torch.tensor(i) for i in input], batch_first=True)

import os

import torch 
from torch.utils.data import DataLoader, Dataset

import conll 

DATAFOLDER = 'augmentation_data/processed_data'



class DeBiasCorefDocumentDataset(Dataset):

    def __init__(self, documentfiles):
        with open(documentfiles, 'r') as f:
            docfiles = f.read().splitlines()
        
        self.documents = [torch.load(os.path.join(DATAFOLDER, f)) for f in docfiles]

        self.all_data = []
        for doc in self.documents:
            self.all_data += doc 
            
        self.total_len = len(self.all_data)
    
    def __len__(self):
        return self.total_len
                
    def __getitem__(self, idx):
        item = self.all_data[idx]
        orig_span1 = torch.tensor([item[0], item[1]])
        orig_span2 = torch.tensor([item[2], item[3]])
        orig_mask = torch.tensor(item[4])
        orig_label = torch.tensor(item[5])
        swap_span1 = torch.tensor([item[6], item[7]])
        swap_span2 = torch.tensor([item[8], item[9]])
        swap_mask = torch.tensor(item[10])
        swap_label = torch.tensor(item[11])
        coref_label = torch.tensor(item[12])
        return (
            orig_span1, orig_span2, orig_mask, orig_label,
            swap_span1, swap_span2, swap_mask, swap_label,
            coref_label
        ) 

def my_collate(batch):
    orig_span1 = torch.stack([b[0] for b in batch])
    orig_span2 = torch.stack([b[1] for b in batch])
    orig_mask = torch.nn.utils.rnn.pad_sequence([b[2] for b in batch], batch_first=True) 
    orig_label = torch.nn.utils.rnn.pad_sequence([b[3] for b in batch], batch_first=True) 
    swap_span1 = torch.stack([b[4] for b in batch])
    swap_span2 = torch.stack([b[5] for b in batch])
    swap_mask = torch.nn.utils.rnn.pad_sequence([b[6] for b in batch], batch_first=True) 
    swap_label = torch.nn.utils.rnn.pad_sequence([b[7] for b in batch], batch_first=True) 
    coref_label = torch.stack([b[8] for b in batch])
    return (
            orig_span1, orig_span2, orig_mask, orig_label,
            swap_span1, swap_span2, swap_mask, swap_label,
            coref_label
        ) 

def get_data_loader(documentsfile):
    dataset = DeBiasCorefDocumentDataset(documentsfile)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=my_collate)
    return dataloader

