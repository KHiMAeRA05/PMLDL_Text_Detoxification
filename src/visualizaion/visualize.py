import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df_generated_paraphrases = pd.read_csv('../../data/interim/predicted.csv')
# Create a list of indices for the sentences
indices = np.arange(10)

# Set the width of the bars
bar_width = 0.3

# Plot the data
plt.figure(figsize=(12, 6))

# Original Toxicity Level
plt.bar(indices, df_generated_paraphrases.iloc[:10]['Original Toxicity Score'], width=bar_width, color='g', align='center', label='Original Toxicity', alpha=0.5)

# Generated Toxicity Level
plt.bar(indices + bar_width, df_generated_paraphrases.iloc[:10]['Generated Toxicity Score'], width=bar_width, color='r', align='center', label='Generated Toxicity', alpha=0.5)

# Similarity
plt.bar(indices + 2*bar_width, df_generated_paraphrases.iloc[:10]['Similarity (meaning)'], width=bar_width, color='b', align='center', label='Similarity', alpha=0.5)

# Add labels
plt.xlabel('Sentence Index')
plt.ylabel('Score')
plt.title('Toxicity and Similarity Comparison')
plt.xticks(indices + bar_width, indices)  # Set x-axis labels to be sentence indices
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()



# Scatter plot comparing similarity and toxicity levels
plt.figure(figsize=(10, 6))

plt.scatter(df_generated_paraphrases['Similarity (meaning)'], df_generated_paraphrases['Original Toxicity Score'], color='blue', label='Original Toxicity', alpha=0.5)
plt.scatter(df_generated_paraphrases['Similarity (meaning)'], df_generated_paraphrases['Generated Toxicity Score'], color='green', label='Generated Toxicity', alpha=0.5)

plt.xlabel('Similarity (meaning)')
plt.ylabel('Toxicity Score')
plt.title('Similarity vs. Toxicity')
plt.legend()
plt.grid(True)

plt.show()



plt.figure(figsize=(10, 6))

plt.boxplot([df_generated_paraphrases['Similarity (meaning)'], df_generated_paraphrases['Original Toxicity Score'], df_generated_paraphrases['Generated Toxicity Score']], labels=['Similarity', 'Original Toxicity', 'Generated Toxicity'])
plt.title('Box Plot of Similarity and Toxicity Scores')
plt.show()




sns.pairplot(df_generated_paraphrases[['Similarity (meaning)', 'Original Toxicity Score', 'Generated Toxicity Score']])
plt.show()