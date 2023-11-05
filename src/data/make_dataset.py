# Prepare dataset
import pandas as pd
df = pd.read_csv('/data/raw/filtered.tsv', sep='\t')
df = df.drop(columns=['Unnamed: 0'])
mask = df['trn_tox'] > df['ref_tox']
# Swap the values of reference and translation, and ref_tox and trn_tox where the condition is True
df.loc[mask, ['reference', 'translation', 'ref_tox', 'trn_tox']] = df.loc[mask, ['translation', 'reference', 'trn_tox', 'ref_tox']].values

df.to_csv('/data/interim/dataset.csv', index=False)  