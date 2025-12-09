#%%
import torch
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import ByteLevelBPETokenizer
import math

# %%
df = pd.read_csv('data.csv')
df = df.drop(["Dataline","PlayerLinenumber","Play","ActSceneLine"], axis=1)
df[['Player', 'PlayerLine']] += '\n'
str_list = df.values.flatten().tolist()[:1000]
str_list = [x for x in str_list if not (isinstance(x, float) and math.isnan(x))]
final_string = ''.join(str_list)
with open('final_training_data.txt', 'w', encoding='utf-8') as f:
    f.write(final_string)

# %%
#naive tokenizer
text_clean = final_string.split()
vocab = sorted(set(text_clean))
vocab_dict = {tok: idx for idx, tok in enumerate(vocab)}
vocab_dict_inv = {idx: tok for idx, tok in enumerate(vocab)}
src = torch.tensor([vocab_dict[x] for x in text_clean])
print(src)

# %%
'''
#later on I will use this
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files= 'final_training_data.txt',vocab_size=32000)
src = tokenizer.encode(final_string)
print(src.ids)
'''

def idx_to_vocab(list_of_idxs):
    list_of_words = [vocab_dict_inv[x] for x in list_of_idxs]
    return list_of_words

def vocab_to_idx(list_of_words):
    list_of_idxs = [vocab_dict[x] for x in list_of_words]
    return list_of_idxs


#%%
batch_size = 12
batch_count = 4
vocab_size = len(vocab_dict)
emb_dim= 32
head_size = 16
src_inputs = src[:-1]
src_targets = src[1:]

#%%
def get_batches(batches_count):
    torch.manual_seed(1)
    all_starts= torch.randint(batch_size, src.shape[0] - batch_size, size=(batches_count, ))
    inputs_batch = torch.zeros(size=[batches_count, batch_size])
    targets_batch = torch.zeros(size=[batches_count, batch_size])
    for start, n in zip(all_starts, range(all_starts.shape[0])):
        inputs_batch[n] = src_inputs[start: start + batch_size]
        targets_batch[n] = src_targets[start: start + batch_size]

    return inputs_batch, targets_batch

inputs, targets = get_batches(batch_count)
inputs = torch._cast_Int(inputs)
targets = torch._cast_Int(targets)

#%%
for inp, tar in zip(inputs, targets):
    n = 0
    while n < inp.shape[0]:
        print(inp.shape)
        print(f"For inputs: {idx_to_vocab(inp[:n + 1].tolist())}")
        print(f"target is:  {idx_to_vocab(tar[n:n+1].tolist())}")
        n+= 1

# %%
emb_table_output = torch.nn.Embedding(num_embeddings=vocab_size,embedding_dim=emb_dim, padding_idx=0)
logits_x = emb_table_output(inputs)
tok_emb = torch.nn.Embedding(num_embeddings=vocab_size,embedding_dim=emb_dim, padding_idx=0)
pos_emb = torch.nn.Embedding(num_embeddings=vocab_size,embedding_dim=emb_dim, padding_idx=0)
B, T, C = logits_x.shape 
x_tok= tok_emb(inputs)
cols = torch.arange(0, batch_count, 1).unsqueeze(1)
rows = torch.arange(0, batch_size, 1).unsqueeze(0)
pos = cols + rows
x_pos= pos_emb(pos)

x = x_tok + x_pos


# %%
