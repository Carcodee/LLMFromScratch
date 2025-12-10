#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def decode(list_of_idxs):
    list_of_words = [vocab_dict_inv[x] for x in list_of_idxs]
    return list_of_words

def encode(list_of_words):
    list_of_idxs = [vocab_dict[x] for x in list_of_words]
    return list_of_idxs


#%%
<<<<<<< HEAD
batch_size = 200
=======
batch_size = 6 
>>>>>>> 52053a2af4ab43abbadbaa1dd62feb372578cded
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
        print(f"For inputs: {decode(inp[:n + 1].tolist())}")
        print(f"target is:  {decode(tar[n:n+1].tolist())}")
        n+= 1

# %%
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(num_embeddings=vocab_size,embedding_dim=emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(num_embeddings=vocab_size,embedding_dim=emb_dim, padding_idx=0)
        self.lm_head = nn.Linear(emb_dim, vocab_size)
    def forward(self, input, target = None):
        B, T = input.shape
        emb_tok = self.tok_emb(input)
        emb_pos = self.pos_emb(torch.arange(T)).unsqueeze(0)
        x = emb_pos + emb_tok
        logits = self.lm_head(x)
        loss = None
        if target is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)    
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
<<<<<<< HEAD

        return logits, loss 
#%%

cTransformer = Transformer()
optimizer = torch.optim.Adam(cTransformer.parameters(), lr=1e-3)
targets = targets.long()

for epoch in range(200):
    optimizer.zero_grad(set_to_none = True)
    y, loss = cTransformer(inputs, targets)
    #print(loss)
    loss.backward()
    optimizer.step()


#%%
def generate(input, max_context_lenght):
    for n in range(max_context_lenght):
        new_logits, _ = cTransformer(input)
        last_tok_logits = new_logits[:, -1, :]
        probs = F.softmax(last_tok_logits, dim=-1)

        new_tok = torch.argmax(probs, dim=-1)
        input = torch.cat((input.squeeze(0), new_tok))
        print(decode(input.tolist()))
        input = input.unsqueeze(0)
        print(input.shape)

input_chain = torch.tensor([[0]])
generate(input_chain, batch_size)
=======

        return logits, loss 
# %%

cTransformer = Transformer()
optimizer = torch.optim.Adam(cTransformer.parameters(), lr=1e-3)
targets = targets.long()

for epoch in range(200):
    optimizer.zero_grad(set_to_none = True)
    y, loss = cTransformer(inputs, targets)
    print(loss)
    loss.backward()
    optimizer.step()

input_chain = torch.tensor([[0]])
next_tok, loss = cTransformer(input_chain)
print(F.softmax(next_tok, dim = 2).max())

>>>>>>> 52053a2af4ab43abbadbaa1dd62feb372578cded


# %%
def generate(input, context_lenght):

    pass
