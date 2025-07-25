from sklearn.base import BaseEstimator, TransformerMixin
import spacy
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import gensim.downloader as gd
import joblib

nlp = spacy.load('en_core_web_sm')

class named_entity_recog_count(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        #initialising an empty set to gather all unique NER labels
        self.unique_ner_labels = set()

        #iterating through the dataframe to collect all unique NER labels
        for _, row in X.iterrows():
            # Extracting text for entity1 and entity2
            entity1_data, entity2_data = row['entity1'], row['entity2']

            #processing text for NER and updating unique_ner_labels with labels from entity1 and entity2
            entity1_doc = nlp(entity1_data)
            entity2_doc = nlp(entity2_data)
            self.unique_ner_labels.update([ent.label_ for ent in entity1_doc.ents])
            self.unique_ner_labels.update([ent.label_ for ent in entity2_doc.ents])

        self.unique_ner_labels = sorted(list(self.unique_ner_labels))
        return self

    def transform(self, X):
        #initialising list to store the feature dictionaries for each record
        ner_features_list = []

        for _, row in X.iterrows():
            #extracting text for entity1 and entity2
            entity1_data, entity2_data = row['entity1'], row['entity2']

            #processing text for NER
            entity1_doc = nlp(entity1_data)
            entity2_doc = nlp(entity2_data)

            #initialising counters for both entity1 and entity2 NER labels
            entity1_ner_count = Counter([ent.label_ for ent in entity1_doc.ents])
            entity2_ner_count = Counter([ent.label_ for ent in entity2_doc.ents])

            #combining counts, adding prefix for entity1 and entity2, and ensuring all labels are included
            combined_counts = {f'H_{label}': entity1_ner_count.get(label, 0) for label in self.unique_ner_labels}
            combined_counts.update({f'T_{label}': entity2_ner_count.get(label, 0) for label in self.unique_ner_labels})

            ner_features_list.append(combined_counts)

        return pd.DataFrame(ner_features_list)


class pos_count(BaseEstimator, TransformerMixin): # A class of POS Tag count for the pipeline
    def fit(self, X, y=None):
        #initialising an empty set to gather all unique POS tags
        self.unique_tags = set()

        #iterating through the dataframe to collect all unique POS tags
        for _, row in X.iterrows():
            text, entity1_span, entity2_span = row['sentence'], row['entity1_span'], row['entity2_span']
            doc = nlp(text)
            #updating unique_tags with tags from the entire text
            self.unique_tags.update([token.pos_ for token in doc])
            #updating unique_tags with tags specifically from entity1 and entity2 sequences
            self.unique_tags.update([doc[i].pos_ for i in entity1_span if i < len(doc)])
            self.unique_tags.update([doc[i].pos_ for i in entity2_span if i < len(doc)])

        self.unique_tags = sorted(list(self.unique_tags))
        return self

    def transform(self, X):
        #initialising list to store the feature dictionaries for each record
        pos_features_list = []

        for _, row in X.iterrows():
            text, entity1_span, entity2_span = row['sentence'], row['entity1_span'], row['entity2_span']
            doc = nlp(text)

            #initializing counters for both entity1 and entity2 sequences
            entity1_pos_count = Counter()
            entity2_pos_count = Counter()

            #counting POS tags for entity1 and entity2 sequences
            for index in entity1_span:
                entity1_pos_count[doc[index].pos_] += 1

            for index in entity2_span:
                entity2_pos_count[doc[index].pos_] += 1

            #combining counts, adding prefix for entity1 and entity2, and ensuring all tags are included
            combined_counts = {f'H_{tag}': entity1_pos_count.get(tag, 0) for tag in self.unique_tags}
            combined_counts.update({f'T_{tag}': entity2_pos_count.get(tag, 0) for tag in self.unique_tags})

            pos_features_list.append(combined_counts)

        return pd.DataFrame(pos_features_list)


class dependency_count(BaseEstimator, TransformerMixin): # A class of dependency count for the pipeline
    def fit(self, X, y=None):
        #initializing an empty set to gather all unique dependency
        unique_deps = set()

        #iterating through the dataframe to collect all unique dependency
        for _, row in X.iterrows():
            text = row['sentence']
            doc = nlp(text)
            #updating unique_deps with all dependency tags from the document
            unique_deps.update([token.dep_ for token in doc])

        self.unique_deps = sorted(list(unique_deps))
        return self

    def transform(self, X):
        #initialising list to store the feature dictionaries for each record
        dep_counts_list = []

        for _, row in X.iterrows():
            text, entity1_span, entity2_span = row['sentence'], row['entity1_span'], row['entity2_span']
            doc = nlp(text)

            #initialising counters for both entity1 and entity2 sequences
            entity1_dep_count = Counter()
            entity2_dep_count = Counter()

            #counting dependency for entity1 and entity2 sequences
            for index in entity1_span:
                entity1_dep_count[doc[index].dep_] += 1

            for index in entity2_span:
                entity2_dep_count[doc[index].dep_] += 1

            #combining counts, ensuring all tags are included even if their count is zero
            combined_counts = {f'H_{dep}': entity1_dep_count.get(dep, 0) for dep in self.unique_deps}
            combined_counts.update({f'T_{dep}': entity2_dep_count.get(dep, 0) for dep in self.unique_deps})

            dep_counts_list.append(combined_counts)

        return pd.DataFrame(dep_counts_list)


class distance(BaseEstimator, TransformerMixin): # A class of distance calculation for the pipeline
    def fit(self, X, y=None):
        #distance does not require learning anything from the training data
        return self  #return self to allow pipeline

    def transform(self, X):
        #initializing an empty list to store the distances
        distances = []

        for _, row in X.iterrows():
            entity1_span, entity2_span = row['entity1_span'], row['entity2_span']

            distance = -1

            if entity1_span and entity2_span:
                #grasp start and end sequence
                entity1_end = max(entity1_span)
                entity2_start = min(entity2_span)
                entity1_start = min(entity1_span)
                entity2_end = max(entity2_span)

                #calculate the distance based on their positions
                if entity1_end < entity2_start:  #wntity1 comes before entity2
                    distance = entity2_start - entity1_end - 1  #subtracting 1 to not count overlapping word
                elif entity2_end < entity1_start:  #entity2 comes before entity1
                    distance = entity1_start - entity2_end - 1  #subtracting 1 to not count overlapping word
                else:  #overlapping
                    distance = 0  #if they overlap, the distance is considered as 0

            distances.append(distance)

        return pd.DataFrame(distances, columns=['distance'])


class word_embedding(BaseEstimator, TransformerMixin): # A class of word embedding transformer for the pipeline
    def __init__(self, word_vectors):
        self.word_vectors = word_vectors

    def get_vector(self, word):
        #return the word vector if it exists, else return a zero vector
        return self.word_vectors[word] if word in self.word_vectors else np.zeros(self.word_vectors.vector_size)

    def get_avg_vector(self, phrase):
        #splitting phrase into words and obtain their vectors
        words = phrase.split()
        vectors = [self.get_vector(word) for word in words]
        #computing the mean of the vectors if the phrase is not empty
        return np.mean(vectors, axis=0) if vectors else np.zeros(self.word_vectors.vector_size)

    def fit(self, X, y=None):
        #word embeddings does not require learning anything from the training data
        return self

    def transform(self, X):
        #initialise an empty list to store word vector
        word_embed = []

        for _, row in X.iterrows():
            head_vector = self.get_avg_vector(row['entity1'])
            tail_vector = self.get_avg_vector(row['entity2'])
            #concatenate the vectors for entity1 and entity2
            combined_vector = np.concatenate([head_vector, tail_vector])
            word_embed.append(combined_vector)

        #convert the list of word embeddings to a DataFrame
        feature_names = [f'embedding_{i}' for i in range(len(word_embed[0]))]
        return pd.DataFrame(word_embed, columns=feature_names)
        
def train_function(X,Y):

    #initiate TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

    #scale using MaxAbsScaler
    scaler = MaxAbsScaler()


    column_transformer = ColumnTransformer([
        ('ner', named_entity_recog_count(), ['entity1', 'entity2']),
        ('pos', pos_count(), ['sentence', 'entity1_span', 'entity2_span']),
        ('dependency', dependency_count(), ['sentence', 'entity1_span', 'entity2_span']),
        ('distance', distance(), ['entity1_span', 'entity2_span']),
        ('word_embedding', word_embedding(word_vectors=gd.load('glove-wiki-gigaword-100')), ['entity1', 'entity2']),
        ('tf-idf', tfidf_vectorizer, 'sentence')
    ], remainder='drop')

   
    model = Pipeline([
        ('feature_extraction', column_transformer),
        ('scale', scaler),
        ('SVC', SVC(kernel='poly', probability=True, random_state=42))

    ])

    model.fit(X, Y)
    joblib.dump(model, "./checkpoints/SVM_POLY_MODEL.pkl")

    return model