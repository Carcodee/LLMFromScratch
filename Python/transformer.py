#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import ByteLevelBPETokenizer
import re
import math

# %%

df = pd.read_csv('data.csv')
df = df.drop(['Dataline','PlayerLinenumber','Play','ActSceneLine'], axis=1)
df[['Player', 'PlayerLine']] += '\n'
str_list = df.values.flatten().tolist()[:10000]
str_list = [x for x in str_list if not (isinstance(x, float) and math.isnan(x))]
final_string = ''.join(str_list)
with open('final_training_data.txt', 'w', encoding='utf-8') as f:
    f.write(final_string)

# %%
df_conversations = pd.read_csv('Conversation.csv')
df_conversations = df_conversations.drop(['Unnamed: 0'], axis=1)
df_conversations['question'] = ('<user>') + df_conversations['question'] + ('</user>\n')
df_conversations['answer'] = ('<bot>') + df_conversations['answer'] + ('</bot>\n')
conv_list = df_conversations.values.flatten().tolist()
conv_list = [x for x in conv_list if not (isinstance(x, float) and math.isnan(x))]
conv_list_str = ''.join(conv_list)
with open('final_post_training_data.txt','w', encoding='utf-8') as f:
    f.write(conv_list_str)

print(conv_list)

# %%
'''
#naive tokenizer
text_clean = re.split(r'(\s+)', final_string)
text_clean_conv = re.split(r'(\s+)', conv_list_str)
vocab = sorted(set(text_clean + text_clean_conv))
vocab_dict = {tok: idx for idx, tok in enumerate(vocab)}
vocab_dict_inv = {idx: tok for idx, tok in enumerate(vocab)}
src = torch.tensor([vocab_dict[x] for x in text_clean])
src_post = torch.tensor([vocab_dict[x] for x in text_clean_conv])
'''

# %%
torch.manual_seed(329846)
device =  'cuda' if torch.cuda.is_available() else 'cpu'

#later on I will use this
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files= 'final_training_data.txt',vocab_size=32000)
encoded_src = tokenizer.encode(final_string)
encoded_src_post = tokenizer.encode(conv_list_str)


# %%
src = torch.tensor(encoded_src.ids, device=device)
src_post = torch.tensor(encoded_src_post.ids, device=device)

train_size = int(src.shape[0] * 0.9)
src_train = src[:train_size]
src_test = src[train_size:]

src_post_train = src_post[:train_size]
src_post_test = src_post[train_size:]
# %%

def decode(list_of_idxs):
#    list_of_words = [vocab_dict_inv[x] for x in list_of_idxs]
    text = tokenizer.decode(list_of_idxs)
    return text

def encode(list_of_words):
#    list_of_idxs = [vocab_dict[x] for x in list_of_words]
    toks = tokenizer.encode(list_of_words)
    return toks


#%%
context_lenght = 256 
pos_emb_lenght = 2048
batch_count = 4 
lr = 1e-3
vocab_size = tokenizer.get_vocab_size()
emb_dim= 512
head_count = 8 
feed_forward = emb_dim * 4

#%%
def get_batches(batches_count, src_data):
    all_starts= torch.randint(0, src_data.shape[0] - context_lenght - 1, size=(batches_count, ))
    inputs = torch.stack([src_data[n : n + context_lenght] for n in all_starts])
    targets = torch.stack([src_data[n + 1: n + context_lenght + 1] for n in all_starts])

    return inputs, targets

#%%

inputs, targets = get_batches(batch_count, src)

inputs = inputs.long().to(device)
targets = targets.long().to(device)
for inp, tar in zip(inputs, targets):
    n = 0
    while n < inp.shape[0]:
        print(f'For inputs: {decode(inp[:n + 1].tolist())}')
        print(f'target is:  {decode(tar[n:n+1].tolist())}')
        n+= 1

#%%
class Head(nn.Module):
    def __init__(self, input_size, head_emb_size):
        super().__init__()
        self.head_emb_size = head_emb_size
        self.to_q = nn.Linear(input_size, head_emb_size)
        self.to_k = nn.Linear(input_size, head_emb_size)
        self.to_v = nn.Linear(input_size, head_emb_size)
    def forward(self, x):
        B, T, C  = x.shape
        Q = self.to_q(x) #(B, T, head_size)
        K = self.to_k(x)
        V = self.to_v(x)
        
        #B, T, C @ B, C, T = B, T, T
        wei = Q @ torch.transpose(K, dim0=-2, dim1=-1)/self.head_emb_size**0.5
        mask = torch.tril(torch.ones(T, T, device=device))
        wei = wei.masked_fill(mask==0, float(-1e9))
        wei = F.softmax(wei, dim=2)
        #B, T, T @ B, T, C = B, T, C
        out = wei @ V
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, head_count, n_emb):
        super().__init__()
        self.head_emb_size = int(n_emb/head_count)
        self.heads = nn.ModuleList([Head(n_emb, self.head_emb_size) for n in range(head_count)])
        self.proj = nn.Linear(n_emb, n_emb)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=2)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )
    #feedforward + residual connection
    def forward(self, x):
        return self.model(x)

class Block(nn.Module):
    def __init__(self, head_count, emb_dim):
        super().__init__()
        self.lyn1 = nn.LayerNorm(emb_dim)
        self.mhatt = MultiHeadAttention(head_count, emb_dim)
        self.lyn2 = nn.LayerNorm(emb_dim)
        self.ff = FeedForward(emb_dim, feed_forward)
    def forward(self, x):
        x = x + self.mhatt(self.lyn1(x))
        x = x + self.ff(self.lyn2(x))
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(num_embeddings=vocab_size,embedding_dim=emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(num_embeddings=pos_emb_lenght,embedding_dim=emb_dim, padding_idx=0)
        self.model_pipeline = nn.Sequential(
            Block(head_count, emb_dim),
            Block(head_count, emb_dim),
            Block(head_count, emb_dim),
            Block(head_count, emb_dim),
            Block(head_count, emb_dim),
            Block(head_count, emb_dim),
        )
        self.lm_head = nn.Linear(emb_dim, vocab_size)
    def forward(self, input, target = None):
        B, T = input.shape
        emb_tok = self.tok_emb(input).to(device)
        emb_pos = self.pos_emb(torch.arange(T, device = device)).unsqueeze(0)
        x = emb_pos + emb_tok
        x = self.model_pipeline(x)
        logits = self.lm_head(x)

        loss = None
        if target is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)    
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)

        return logits, loss 
#%%
cTransformer = Transformer().to(device=device)
optimizer = torch.optim.Adam(cTransformer.parameters(), lr=lr)

#%%
cTransformer.load_state_dict(torch.load('checkpoint.pth'))

#%%
def Train(epochs, src_data_train, src_data_test):
    for n in range(epochs):
        optimizer.zero_grad(set_to_none = True)
        train_x, train_y = get_batches(batch_count, src_data_train)
        train_x = train_x.long().to(device)
        train_y = train_y.long().to(device)
        y, loss = cTransformer(train_x, train_y)
        loss.backward()
        optimizer.step()
        if n % 50 == 0:
            with torch.no_grad():
                test_x, test_y = get_batches(batch_count, src_data_test)
                test_x = test_x.long().to(device)
                test_y = test_y.long().to(device)
                _, loss_test = cTransformer(test_x, test_y)
                print(f'train: {n}: {loss}, test: {n}: {loss_test}')


#%%
epoch_counter = 0
Train(5000, src_train, src_test)
epoch_counter+= 500


#%%
Train(2500, src_post_train, src_post_test)
epoch_counter+= 1000

torch.save({'model': cTransformer.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch_counter},
            'checkpoint.pth')

#%%
torch.save(cTransformer.cpu().state_dict(), 'model.pth')
cTransformer.to(device)

#%%
def generate(input, max_size):
    generated_text = ''
    seq = torch.tensor([], dtype = torch.long).to(device)
    seq = torch.cat((seq, input.squeeze()), dim=0)
    for n in range(max_size):
        input = input[:,-context_lenght:] 
        new_logits, _ = cTransformer(input)
        last_tok_logits = new_logits[:, -1, :]
        probs = F.softmax(last_tok_logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)  
        input = torch.cat((input, next_tok), dim=1)
        seq = torch.cat((seq, input.squeeze()[input.shape[1]-1:]), dim=0)
    generated_text = ''.join(decode(seq.tolist()))
    print(generated_text)
    with open('output_file.txt', 'w', encoding='utf-8') as f:
        f.write(generated_text)


def Prompt(text):
    text = f"<user>{text}</user>"
    toks = encode(text)
    tokens = torch.tensor([toks.ids]).to(device)
    generate(tokens, 10000)

#%%
state = torch.load('model.pth', map_location='cuda')
cTransformer.load_state_dict(state)
cTransformer.eval()
#%%
Prompt('how are you doing?')


# %%
