import numpy as np
import argparse
from tqdm import tqdm
import pandas as pd
import os
import pickle
import pandas as pd


parser = argparse.ArgumentParser(description='Create literals')
parser.add_argument('--dataset', default='new_t3', metavar='',
                    help='which dataset in {`YAGO3-10`, `FB15k`, `FB15k-237`} to be used? (default: YAGO3-10)')
args = parser.parse_args()


train=pd.read_csv(f'../datasets/{args.dataset}/train.txt', sep="\t",header=None,names=['head','relation','tail'])
valid=pd.read_csv(f'../datasets/{args.dataset}/valid.txt', sep="\t",header=None,names=['head','relation','tail'])
test=pd.read_csv(f'../datasets/{args.dataset}/test.txt', sep="\t",header=None,names=['head','relation','tail'])
all_df=pd.concat([train, valid, test], ignore_index=True)
#print(all_df)
print("# of Triplets", len(all_df))

# Entity dictionary
entities=set(all_df['head'].unique()) | set(all_df['tail'].unique())
entity_dict = {v: k for k, v in enumerate(entities)}

print("# of Entites:", len(entity_dict))
with open(f'../datasets/{args.dataset}/entities.dict','wb') as fw:
    pickle.dump(entity_dict, fw)


# Relation dictionary
relations=all_df['relation'].unique()
relation_dict = {v: k for k, v in enumerate(relations)}

print("# of Entity Relations:",len(relation_dict))
with open(f'../datasets/{args.dataset}/relations.dict','wb') as fw:
    pickle.dump(relation_dict, fw)



# Load raw literals
df = pd.read_csv(f'../datasets/{args.dataset}/literals/numerical_literals.txt', header=None, sep='\t')
numrel_dict = {v: k for k, v in enumerate(df[1].unique())}
print("# of Numeric Relations:", len(numrel_dict))

# Resulting file
num_lit = np.zeros([len(entity_dict), len(numrel_dict)], dtype=np.float32)


# Create literal wrt vocab
for i, (s, p, lit) in enumerate(df.values):
    try:

        if "id" in p:
            num_lit[entity_dict[s.lower()], numrel_dict[p]] = 1.0
        else:
            num_lit[entity_dict[s.lower()], numrel_dict[p]] = lit
    except KeyError:
        continue

np.save(f'../datasets/{args.dataset}/literals/numerical_literals.npy', num_lit)



