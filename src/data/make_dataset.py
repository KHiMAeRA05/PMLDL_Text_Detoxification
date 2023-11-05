# Prepare dataset
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, DistilBertTokenizer, DistilBertModel, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import torch
from sklearn.metrics.pairwise import cosine_similarity

# ------------------

# Init functions from predict_model.py (I don't know how to make modules....)

class TextSimilarityModel:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(model_name)

    def calculate_similarity(self, text1, text2):
        input_ids1 = self.tokenizer(text1, return_tensors='pt').input_ids
        input_ids2 = self.tokenizer(text2, return_tensors='pt').input_ids
        with torch.no_grad():
            emb1 = self.model(input_ids1).last_hidden_state[:, 0, :]
            emb2 = self.model(input_ids2).last_hidden_state[:, 0, :]
        return cosine_similarity(emb1, emb2)

    def predict(self, text1, text2):
        similarity_score = self.calculate_similarity(text1, text2)[0][0]
        return similarity_score

class ToxicityPredictor:
    def __init__(self, model_name='SkolkovoInstitute/roberta_toxicity_classifier'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)

    def predict(self, text):
        result = self.classifier(text)
        score = result[0]['score']
        if result[0]['label'] == 'neutral':
            score = 1 - score
        return score

simularity_model = TextSimilarityModel()
toxicity_model = ToxicityPredictor()

# Define a function to calculate similarity
def calculate_similarity(sentence1, sentence2):
    #print(sentence1, sentence2)
    return simularity_model.predict(text1=sentence1, text2=sentence2)

# Define a function to assess toxicity (if you have a toxicity classifier)
def assess_toxicity(sentence):
    return toxicity_model.predict(sentence)
  
# ------------------

# Data preprocessing
df = pd.read_csv('../../data/raw/filtered.tsv', sep='\t')
df = df.drop(columns=['Unnamed: 0'])

df['similarity'] = [calculate_similarity(row['reference'], row['translation']) for _, row in tqdm(df.iterrows(), total=len(df['reference']))]
df['ref_tox'] = [assess_toxicity(row['reference']) for _, row in df.iterrows()]
df['trn_tox'] = [assess_toxicity(row['translation']) for _, row in df.iterrows()]

mask = df['trn_tox'] > df['ref_tox']
# Swap the values of reference and translation, and ref_tox and trn_tox where the condition is True
df.loc[mask, ['reference', 'translation', 'ref_tox', 'trn_tox']] = df.loc[mask, ['translation', 'reference', 'trn_tox', 'ref_tox']].values

df.to_csv('../../data/interim/dataset.csv', index=False)  
