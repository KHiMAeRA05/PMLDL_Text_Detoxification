import pandas as pd
from tqdm import tqdm_notebook
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer, util

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

df_toxicity = pd.read_csv("\..\..\data\external\toxicity_en.csv")
df_toxicity = df_toxicity[df_toxicity['is_toxic'] == 'Toxic']


simularity_model = TextSimilarityModel()
toxicity_model = ToxicityPredictor()

# Define a function to calculate similarity
def calculate_similarity(sentence1, sentence2):
    #print(sentence1, sentence2)
    return simularity_model.predict(text1=sentence1, text2=sentence2)

# Define a function to assess toxicity (if you have a toxicity classifier)
def assess_toxicity(sentence):
    return toxicity_model.predict(sentence)

# Generate paraphrases and calculate similarity
generated_paraphrases = []
texts=df_toxicity['text'].to_list()
encoded_texts = tokenizer(texts, return_tensors="pt", padding=True)
encoded_texts = {k: v.to(trainer.model.device) for k,v in encoded_texts.items()}
outputs = trainer.model.generate(encoded_texts["input_ids"])

for output, sentence in tqdm_notebook(zip(outputs, texts), total=len(texts)):
    paraphrase = tokenizer.decode(output, skip_special_tokens=True)
    similarity = calculate_similarity(sentence, paraphrase)
    original_toxicity_score = assess_toxicity(sentence)
    generated_toxicity_score = assess_toxicity(paraphrase)
    generated_paraphrases.append({
        'Original Sentence': sentence,
        'Generated Paraphrase': paraphrase,
        'Similarity (meaning)': similarity,
        'Original Toxicity Score': original_toxicity_score,
        'Generated Toxicity Score': generated_toxicity_score
    })

# Create a DataFrame
df_generated_paraphrases = pd.DataFrame(generated_paraphrases)

df_generated_paraphrases.to_csv('../../data/interim/predicted.csv', index=False)  