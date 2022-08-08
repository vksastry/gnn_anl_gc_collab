"""Download the QM9 dataset from GitHub, parse into train/test/validation sets"""
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import json

data = pd.read_json('https://github.com/globus-labs/g4mp2-atomization-energy/raw/master/data/output/g4mp2_data.json.gz',
                   lines=True)
print(f'Downloaded {len(data)} training entries')

test_set = data.query('in_holdout')
print(f'Set aside {len(test_set)} training entries')

train_set, val_set = train_test_split(data.query('not in_holdout'), test_size=0.1, random_state=1)
print(f'Split off {len(train_set)} training and {len(val_set)} validation entries')

out_dir = Path('data') / 'qm9'
out_dir.mkdir(exist_ok=True)

for name, dataset in zip(['train', 'valid', 'test'], [train_set, val_set, test_set]):
    dataset = dataset.sample(frac=1.)  # Shuffle contents
    dataset.rename(columns={'smiles_0': 'smiles', 'g4mp2_atom': 'output'})[['smiles', 'output']].to_csv(out_dir / f'{name}.csv', index=False)

