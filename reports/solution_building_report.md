# Baseline: Dictionary based
As an initial baseline, I used a pre-trained model for text paraphrase. The pre-trained models were also used to assess the toxicity of the text and the similarity of the original and generated texts. 
This provided a basic level of detoxification, but the results were limited.

# Hypothesis 1: NLP model
Since this is a paraphrase task, some appropriate NLP model should probably be used here.

# Hypothesis 2: Pre-trained model
It would be best to use an already pre-trained model since I don't have enough resources to implement it.

# Hypothesis 3: Pretrained embeddings
The relevance of the model has to be evaluated somehow. The generated text should be checked not only for the level of toxicity, but also for the level of similarity with the original phrase. 
For these tasks, I think it would be better to find ready-made pre-trained models.

# Results
The final solution combines the pre-trained models, resulting in a highly effective text detoxification model. 
Further details can be found in the Final Solution Report.
