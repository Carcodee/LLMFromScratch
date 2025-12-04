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
vocab_inv = dict(zip(vocab_dict.values(), vocab_dict.keys())) 
src = torch.tensor([vocab_dict[x] for x in text_clean])
print(src)

# %%
#later on I will use this
#tokenizer = ByteLevelBPETokenizer()
#tokenizer.train(files= 'final_training_data.txt',vocab_size=32000)
#src = tokenizer.encode(final_string)
#print(src.ids)


#%%
inputs = src[:-1]
targets = src[1:]

print([vocab[x] for x in inputs])
print([vocab[x] for x in targets])
print([vocab_inv[x.item()] for x in inputs])
print([vocab_inv[x.item()] for x in targets])


# %%
