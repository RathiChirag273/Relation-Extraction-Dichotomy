# imports
from pathlib import Path
import os
import pickle
from tqdm import tqdm

# as pytorch dataset
import torch
from torch.utils.data import Dataset, DataLoader
import itertools
import random

class DocREDDataset(Dataset):
    def __init__(self, data, na_factor=1):
        self.dataset = data
        self.xs = []
        self.ys = []
        self.dists = []

        # foreach document in dataset
        for idx in tqdm(range(len(self.dataset))):
            data_i = self.dataset[idx]
            ebd = data_i['embed']
            rs = data_i['rs']
            refs = data_i['ner_ref']
            # foreach relation label in the document, generate inputs and class labels
            for r in rs:
                # foreach relatio in sample
                idx_h = r['h']
                idx_t = r['t']
                y = r['r']

                # coref
                ref_h, ref_h_type = refs[idx_h]
                ref_t, ref_t_type = refs[idx_t]
                coref = (ref_h + ref_t)
                coref_type = (ref_h_type + ref_t_type)
                coref_type = (ref_h_type + ref_t_type)
                pad_size = ebd.shape[0] - coref.shape[0]
                coref_padded, coref_type_padded = coref, coref_type
                if pad_size > 0:
                    coref_padded = torch.cat([coref, torch.zeros(pad_size, coref.shape[1])], dim=0)
                    coref_type_padded = torch.cat([coref_type, torch.zeros(pad_size, coref.shape[1])], dim=0)
                    ref_h = torch.cat([ref_h, torch.zeros(pad_size, coref.shape[1])], dim=0)
                    ref_t = torch.cat([ref_t, torch.zeros(pad_size, coref.shape[1])], dim=0)
                

                x = torch.cat([ebd, coref_padded, coref_type_padded, ref_h, ref_t], dim=1)
                self.xs.append(x)
                self.ys.append(y)

            # NA relations
            # add up to len(rs) na examples
            count_na = 0
            idxs_ht = []
            for idx_h, (entity, _) in enumerate(data_i['ner_set']):
                for idx_t, (target, _) in enumerate([e for e in data_i['ner_set'] if e != entity]):
                    # check if relation exists
                    relation_exists = False
                    for r in rs:
                        if idx_h == r['h'] and idx_t == r['t']:
                            relation_exists = True
                            break
                    if relation_exists:
                        continue
                    idxs_ht.append((idx_h, idx_t))
            # randomly select from the list of NA
            # print(idxs_ht)
            idxs_na = random.sample(idxs_ht, k=int(len(rs)*na_factor))
            for idx_h, idx_t in idxs_na:

                # realtion does not exist, add NA relation
                ref_h, ref_h_type = refs[idx_h]
                ref_t, ref_t_type = refs[idx_t]
                coref = (ref_h + ref_t)
                coref_type = (ref_h_type + ref_t_type)
                pad_size = ebd.shape[0] - coref.shape[0]
                coref_padded, coref_type_padded = coref, coref_type
                if pad_size > 0:
                    coref_padded = torch.cat([coref, torch.zeros(pad_size, coref.shape[1])], dim=0)
                    coref_type_padded = torch.cat([coref_type, torch.zeros(pad_size, coref_type.shape[1])], dim=0)
                    ref_h = torch.cat([ref_h, torch.zeros(pad_size, coref.shape[1])], dim=0)
                    ref_t = torch.cat([ref_t, torch.zeros(pad_size, coref.shape[1])], dim=0)
                x = torch.cat([ebd, coref_padded, coref_type_padded, ref_h, ref_t], dim=1)
                self.xs.append(x)
                y = torch.zeros(97)
                y[0] = 1.0
                self.ys.append(y)
                count_na += 1
            
        
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]
    
# Collate function to pad sequences
def collate_fn(batch):
    x, y = zip(*batch)  # Unpack batch

    # Pad sequences to the max length in batch
    pad_embed = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0.0)
    y = torch.stack(y)
    return pad_embed, y # Return padded tensors and lengths