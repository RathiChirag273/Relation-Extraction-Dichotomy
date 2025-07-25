import pandas as pd
import re
import json

class DOCRED_Processing:
    def __init__(self, data, is_input = False):
        self.df = self.convert_docred_to_dataframe(data, is_input)
        self.df['entity1_span'] = self.df.apply(lambda row: self.find_entity_indices(row, 1), axis=1)
        self.df['entity2_span'] = self.df.apply(lambda row: self.find_entity_indices(row, 2), axis=1)
        self.df = self.df.drop(index=[index for index, row in self.df.iterrows() if row["entity1_span"] == []])
        self.df = self.df.drop(index=[index for index, row in self.df.iterrows() if row["entity2_span"] == []])

    def convert_docred_to_dataframe(self, data, is_input):
        
        if is_input == False:
            pro_data = []
            with open('./dataset_DocRED/rel_info.json', 'rb') as f:
                id2rel = json.load(f)

            for doc in data:
                title = doc["title"]
                sentences = doc["sents"]
                vertex_set = doc["vertexSet"]
                labels = doc.get("labels", [])  # Some sets may not have labels

                for label in labels:
                    h_idx, t_idx, relation, evidence = label["h"], label["t"], label["r"], label["evidence"]

                    # Find mentions of head and tail entities that appear in the evidence sentences
                    head_mentions = [m for m in vertex_set[h_idx] if m["sent_id"] in evidence]
                    tail_mentions = [m for m in vertex_set[t_idx] if m["sent_id"] in evidence]

                    # If multiple mentions exist, take the first ones in the evidence sentences
                    if head_mentions and tail_mentions:
                        head_entity = head_mentions[0]
                        tail_entity = tail_mentions[0]
                    else:
                        # Default to the first mention if none are explicitly in evidence
                        head_entity = vertex_set[h_idx][0]
                        tail_entity = vertex_set[t_idx][0]

                    if len(evidence) > 1:
                        sen = ""
                        sent_tokens = []
                        for i in evidence:
                            tokens = sentences[i]
                            sentence = " ".join(sentences[i])
                            sen = sen + " " + sentence
                            sent_tokens = sent_tokens + tokens
                    else:
                        sentence_id = head_entity["sent_id"]  # Assume head entity sentence as main sentence
                        sen = " ".join(sentences[sentence_id])
                        sent_tokens = sentences[sentence_id]

                    pro_data.append({
                        "entity1": head_entity["name"],
                        "entity2": tail_entity["name"],
                        "original_doc": title,
                        "sent_tokens": sent_tokens,
                        "sentence": sen,
                        "relation": id2rel[relation]
                    })
                
            return pd.DataFrame(pro_data)
        
        else:
            return pd.DataFrame(data)
    
    def find_entity_indices(self, row, entity_num):
        def remove_punctuation(word):
            return re.sub(r'[^\w\s]', '', word)  # Remove punctuation

        sentence_tokens = row['sent_tokens']
        entity = row[f'entity{entity_num}']

        words = entity.split()  # Split search string into words
        indices = []
        seen_words = set()  # Track already added words

        for word in words:
            clean_word = remove_punctuation(word)  # Normalize word
            if clean_word not in seen_words:
                for i, w in enumerate(sentence_tokens):
                    clean_w = remove_punctuation(w)  # Normalize sentence token
                    if clean_w == clean_word:
                        indices.append(i)
                        seen_words.add(clean_word)
                        break
        
        return indices
