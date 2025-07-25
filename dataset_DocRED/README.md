# DocRED

In this directory you should have the following files from the `DocRED` dataset:

```raw
.
├── DocRED_baseline_metadata/
│   ├── char_vec.npy
│   ├── char2id.json
│   ├── ner2id.json
│   ├── rel2id.json
│   ├── vec.npy
│   └── word2id.json
├── dev.json
├── rel_info.json
├── test.json
└── train_annotated.json
```

Available at [google drive](https://drive.google.com/drive/folders/1_sTe7NTio8FLfJQdBeO-N5G9o2hEZ6Nd?usp=sharing)  

## Notes  

### 1. Annotated Data  

The following annotated datasets are stored in raw json format:  

- `dev.json` - Development dataset?
- `test.json` - Test dataset
- `train_annotated.json` - Training dataset (manually labeled)  

As explained by the creator of `DocRED` at <https://github.com/thunlp/DocRED/tree/master/data/README.md>, the structure of the datasets are as follows:  

```raw
Data Format:
{
  'title',
  'sents':     [
                  [word in sent 0],
                  [word in sent 1]
               ]
  'vertexSet': [
                  [
                    { 'name': mention_name, 
                      'sent_id': mention in which sentence, 
                      'pos': postion of mention in a sentence, 
                      'type': NER_type}
                    {another mention}
                  ], 
                  [another entity]
                ]
  'labels':   [
                {
                  'h': idx of head entity in vertexSet,
                  't': idx of tail entity in vertexSet,
                  'r': relation,
                  'evidence': evidence sentences' id
                }
              ]
}
```

### 2. Auxiliary Information  

- `rel_info.json` - Table for mapping labels to types of relation in natural language.  

### 3. Word/Character Embeddings  

Located in the `DocRED_baseline_metadata` subdirectory are a set of pre-trained word embeddings used by the baseline models provided by `DocRED`.  

Usage (with `PyTorch`):  

1. Load Embeddings table `vec.npy` to `nn.Embedding` using `nn.Embedding.weight.data.copy_()`
2. Map word to index from `word2id.json` (done in pre-processing)
3. Use index to retrieve word embedding in training.

**Note**:  
In addition to word embeddings, in RE, **Named Entity** embeddings and **Coreference** embeddings are also useful. Those are learned as trainable parameters (i.e. for typed named entities, coref clusters).  
