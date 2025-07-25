import json
from pathlib import Path
import numpy as np
import os
import glob
from itertools import chain

# from transformers import BertTokenizer, BertModel
import transformers
from transformers import AutoTokenizer, DistilBertModel 
import torch
import pickle
from tqdm import tqdm


DIR_OUT_DEFAULT = Path(__file__).parent.absolute().parent/"dataset_prep"
DIR_DATA_DEFAULT = Path(__file__).parent.absolute().parent/"dataset_DocRED"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO : Implement preprocessing scripts
"""
Generate document-level embeddings

Parameters
----------
data : array_like
    loaded from json dataset
tokenizer : transformers.Tokenizer
model_embed : embedding model
"""
def prepDocRED(data, tokenizer:transformers.AutoTokenizer, model_embed:DistilBertModel, path_out=DIR_OUT_DEFAULT/'data.pickle'):
    data_prep = [
        {
            'sents': None, 
            'embed': None, # Text embeddings
            'ner': [], # named entities
            'ner_ref': [], # named entitiy references, tuple(pos, type)
            'ner_set': [], # named entitiy set, tuple(name, type)
            'rs':[] # relations
        } for doc in data
    ]

    # load in dataset metadata
    fN2id = open(DIR_DATA_DEFAULT/'DocRED_baseline_metadata'/'ner2id.json', 'r', encoding='utf-8')
    fR2id = open(DIR_DATA_DEFAULT/'DocRED_baseline_metadata'/'rel2id.json', 'r', encoding='utf-8')
    n2id = json.load(fN2id)
    fN2id.close()
    r2id = json.load(fR2id)
    fR2id.close()

    # foreach document in dataset
    pBar = tqdm(desc=f"doc", total=len(data))
    for idx_doc, doc in enumerate(data):
        # prep text embeddings (document-level)
        sentences = list(chain.from_iterable(doc['sents'])) # Concatenate sentences 
        data_prep[idx_doc]['sents'] = sentences
        inputs = tokenizer(sentences, padding=False, truncation=True, return_tensors="pt", is_split_into_words=True, 
                           max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        data_prep[idx_doc]['embed'] = outputs.last_hidden_state.cpu().squeeze(0)
        len_embed = data_prep[idx_doc]['embed'].shape[0]

        # prep ner
        for entities in doc['vertexSet']:
            data_prep[idx_doc]['ner_set'].append((entities[0]['name'], entities[0]['type']))
            # NER mentions
            mentions = []
            # foreach type of entity
            for entity in entities:
                # foreach entity instance
                idx_start, idx_end = entity['pos'] 
                idx_sent = entity['sent_id']
                len_before = len(list(chain.from_iterable(doc['sents'][:idx_sent])))
                idx_start += len_before
                idx_end += len_before
                # preped NER label 
                e = {
                    'name': entity['name'], 
                    'pos': (idx_start, idx_end), 
                    'type': n2id[entity['type']]
                }
                mentions.append(e)
            data_prep[idx_doc]['ner'].append(mentions)

            # references
            ref = torch.zeros((len_embed, 1), dtype=torch.float16)
            ref_type = torch.zeros((len_embed, 1), dtype=torch.float16)
            for mention in mentions:
                idx_start, idx_end = mention['pos']
                ref[idx_start:idx_end] = 1.0
                ref_type[idx_start:idx_end] = n2id[entities[0]['type']]
            data_prep[idx_doc]['ner_ref'].append((ref, ref_type))

        # prep relations label
        for label in doc['labels']:
            idx_h = label['h']
            idx_t = label['t']
            id_r = r2id[label['r']] # relation id, 0 == Na

            # id_h = n2id[doc['vertexSet'][idx_h][0]['type']]
            # id_t = n2id[doc['vertexSet'][idx_t][0]['type']]

            # on-hot encoding for relation type
            r_encode = torch.zeros(97)
            r_encode[id_r] = 1

            l = {
                'h': idx_h,
                't': idx_t, 
                'r': r_encode 
            }
            data_prep[idx_doc]['rs'].append(l)
        pBar.update()
    pBar.close()

    # save to out path
    fOut = open(path_out, 'wb')
    pickle.dump(data_prep, fOut)
    fOut.close()

    pass


if __name__ == "__main__":
    DIR_OUT_DEFAULT.mkdir(exist_ok=1)
    for f in glob.glob(f'{DIR_OUT_DEFAULT}/*'):
        os.remove(f)

    # load model and tokenizer
    # with flash attention 2, see https://huggingface.co/docs/transformers/model_doc/distilbert
    tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
    model = DistilBertModel.from_pretrained("distilbert/distilbert-base-uncased", torch_dtype=torch.float16, attn_implementation="sdpa", device_map="cuda:0")
    model = model.to(device)

    dir_data = Path(__file__).parent.absolute().parent/"dataset_DocRED"
    fname_data = ["dev", "train_annotated"]
    
    for fname in fname_data:
        fData = open(dir_data/f'{fname}.json'.__str__(), 'rb')
        data = json.load(fData)
        fData.close()

        prepDocRED(data=data, tokenizer=tokenizer, model_embed=model, path_out=DIR_OUT_DEFAULT/f'{fname}_prep.pickle')
    

    exit(0)
