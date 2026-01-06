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
import matplotlib.pyplot as plt
from datetime import datetime

# %%

df = pd.read_csv('./text_data/data.csv')
df = df.drop(['Dataline','PlayerLinenumber','Play','ActSceneLine'], axis=1)
df[['Player', 'PlayerLine']] += '\n'
str_list = df.values.flatten().tolist()[:10000]
str_list = [x for x in str_list if not (isinstance(x, float) and math.isnan(x))]
final_string = ''.join(str_list)
with open('./text_data/final_training_data.txt', 'w', encoding='utf-8') as f:
    f.write(final_string)

# %%
'''
df_conversations = pd.read_csv('./text_data/Conversation.csv', nrows=10000)
df_conversations = df_conversations.drop(['conversation_id', 'turn', 'intent'], axis=1)
df_conversations_new = '<' + df_conversations['role'] + '>' +  df_conversations['message'] + '</' + df_conversations['role'] + '>\n'
conv_list = df_conversations_new.values.flatten().tolist()
'''

#%%
df_conversations = pd.read_csv('./text_data/Conversation.csv', nrows=10000)
df_conversations = df_conversations.drop(['Unnamed: 0', 'question'], axis=1)
df_conversations_new = '<user>' +  df_conversations['answer'] + '</user>\n'
conv_list = df_conversations_new.values.flatten().tolist()

#%%
conv_list_str = ''.join(conv_list)
#%%
with open('./text_data/final_post_training_data.txt','w', encoding='utf-8') as f:
    f.write(conv_list_str)

with open('./text_data/vocab_training.txt','w', encoding='utf-8') as f:
    f.write(final_string)

with open('./text_data/vocab_training.txt','a', encoding='utf-8') as f:
    f.write(conv_list_str)

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

#later on I will use this
special_tokens = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<user>",
    "</user>",
    "<bot>",
    "</bot>"
]
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files= './text_data/vocab_training.txt',vocab_size=32000, special_tokens = special_tokens)
encoded_src = tokenizer.encode(final_string)
encoded_src_post = tokenizer.encode(conv_list_str)



# %%
device =  'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(329846)
if torch.cuda.is_available():
    torch.cuda.manual_seed(329846)
torch.set_float32_matmul_precision('high')
context_lenght = 128
pos_emb_lenght = 2048
batch_count =  64
lr = 1e-3
vocab_size = tokenizer.get_vocab_size()
emb_dim= 512
head_count = 8 
feed_forward = emb_dim * 4
temperature = 1

# %%
src = torch.tensor(encoded_src.ids, device=device)
src_post = torch.tensor(encoded_src_post.ids, device=device)

train_size = int(src.shape[0] * 0.9)
train_size_post = int(src_post.shape[0] * 0.9)
src_train = src[:train_size]
src_test = src[train_size:]

src_post_train = src_post[:train_size_post]
src_post_test = src_post[train_size_post:]
print(src_post.shape)
print(src_post_train.shape)
print(train_size)
print(tokenizer.decode(src_post_test[:1000].tolist()))
# %%

def decode(list_of_idxs):
#    list_of_words = [vocab_dict_inv[x] for x in list_of_idxs]
    text = tokenizer.decode(list_of_idxs, skip_special_tokens=False)
    return text

def encode(list_of_words):
#    list_of_idxs = [vocab_dict[x] for x in list_of_words]
    toks = tokenizer.encode(list_of_words)
    return toks


#%%
def get_batches(batches_count, src_data):
    src_usable = src_data[: src_data.shape[0] - (src_data.shape[0] % context_lenght) + 1]
    inputs =src_usable[:-1].view(-1, context_lenght)
    targets =src_usable[1:].view(-1, context_lenght)
    all_starts= torch.randint(0, inputs.shape[0], size=(batches_count, ), dtype=torch.long).to(device=device)

    inputs = inputs[all_starts]
    targets = targets[all_starts]
    return inputs, targets


#%%

inputs, targets = get_batches(batch_count, src_post_train)
print(inputs.shape)

'''
for inp, tar in zip(inputs, targets):
    n = 0
    while n < inp.shape[0]:
        print(f'For inputs: {decode(inp[:n + 1].tolist())}')
        print(f'target is:  {decode(tar[n:n+1].tolist())}')
        n+= 1
'''
#%%
class Head(nn.Module):
    def __init__(self, input_size, head_emb_size):
        super().__init__()
        self.head_emb_size = head_emb_size
        self.to_q = nn.Linear(input_size, head_emb_size)
        self.to_k = nn.Linear(input_size, head_emb_size)
        self.to_v = nn.Linear(input_size, head_emb_size)
        self.register_buffer("mask", torch.tril(torch.ones(context_lenght, context_lenght)))
    def forward(self, x):
        B, T, C  = x.shape
        Q = self.to_q(x) #(B, T, head_size)
        K = self.to_k(x)
        V = self.to_v(x)
        
        #B, T, C @ B, C, T = B, T, T
        wei = Q @ torch.transpose(K, dim0=-2, dim1=-1)/self.head_emb_size**0.5
        mask = self.mask[:T, :T]
        wei = wei.masked_fill(mask==0, float(-1e9))
        wei = F.softmax(wei, dim=2)
        wei = F.dropout(wei, p=0.4, training=self.training)
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
        out = F.dropout(self.proj(out), p=0.4, training=self.training)
        return out


class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, input_size),
            nn.Dropout(0.4),
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
        self.dropout = nn.Dropout(0.4)
    def forward(self, x):
        x = x + self.dropout(self.mhatt(self.lyn1(x)))
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
        self.register_buffer(
            "pos_ids",
            torch.arange(pos_emb_lenght)
        )
        
        self.ln_f = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)
    def forward(self, input, target = None):
        B, T = input.shape
        emb_tok = self.tok_emb(input).to(device)
        pos = self.pos_ids[:T].unsqueeze(0)
        emb_pos = self.pos_emb(pos)
        x = emb_pos + emb_tok
        x = F.dropout(x, p=0.4, training=self.training)
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
optimizer = torch.optim.AdamW(cTransformer.parameters(), lr=lr)
param_counter = 0
#for p in cTransformer.parameters():
#    p+= p.numel()

#print(param_counter)

cTransformer = torch.compile(cTransformer, backend='eager')

#%%
#cTransformer.load_state_dict(torch.load('checkpoint.pth'))

#%%
def Train(epochs, src_data_train, src_data_test, print_ts = False):
    losses = []
    for n in range(epochs):
        optimizer.zero_grad(set_to_none = True)
        train_x, train_y = get_batches(batch_count, src_data_train)
        train_x = train_x.long().to(device)
        train_y = train_y.long().to(device)
        y, loss = cTransformer(train_x, train_y)
        loss.backward()
        optimizer.step()
        if n % 20 == 0:
            with torch.no_grad():

                if print_ts:
                    time_start = datetime.now().microsecond
                test_x, test_y = get_batches(batch_count, src_data_test)
                test_x = test_x.long().to(device)
                test_y = test_y.long().to(device)
                _, loss_test = cTransformer(test_x, test_y)
                losses.append((loss, loss_test))
                output = f'train: {n}: {loss:.4f}, test: {n}: {loss_test:.4f}'
                if print_ts:
                    torch.cuda.synchronize()
                    time_end = datetime.now().microsecond
                    #ms
                    dt = time_end - time_start 
                    output += f', time {dt}ms'
                print(output)
     
        
    return losses

#%%
epoch_counter = 0
losses = Train(1000, src_train, src_test)
epoch_counter+=1000 


#%%
losses_post = Train(1000, src_post_train, src_post_test, True)
epoch_counter+= 1000

#%%
torch.save({'model': cTransformer.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch_counter},
            'checkpoint.pth')
#%%
losses_train = []
losses_val = []
for loss in losses_post:
    losses_train.append(loss[0].detach().cpu().item())
    losses_val.append(loss[1].detach().cpu().item())


print(losses_train)
plt.figure()
plt.plot(losses_train, label = "train loss")
plt.plot(losses_val, label = "val loss")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.grid(True)
plt.legend()
plt.show()
#%%
torch.save(cTransformer.cpu().state_dict(), 'model.pth')
cTransformer.to(device)

#%%
def generate(prompt_tokens, max_size):
    seq = prompt_tokens.clone()   # full growing sequence
    generated_text = ""

    for _ in range(max_size):

        model_in = seq[:, -context_lenght:]

        logits, _ = cTransformer(model_in)
        last_logits = logits[:, -1, :]
        topk = torch.topk(last_logits, 50)
        topk_logits = topk.values
        topk_indices = topk.indices
        probs = F.softmax(topk_logits/temperature, dim=-1)
        probs_ploted = probs.detach().cpu().clone()
        probs_ploted = probs_ploted.squeeze(dim=0)
        #plt.plot(probs_ploted.tolist())
        #plt.show()


        next_tok = topk_indices.gather(-1, 
                            torch.multinomial(probs, 1))

        seq = torch.cat((seq, next_tok), dim=1)

        recent = decode(seq[0, -20:].tolist())
        generated_text = decode(seq[0].tolist())
#        print(generated_text)

        if "</user>" in recent:
            break

    with open("output_file.txt", "w", encoding="utf-8") as f:
        f.write(generated_text)

    return generated_text
def Prompt(text):
    text = f"{text}"
    toks = encode(text)
    tokens = torch.tensor([toks.ids]).to(device)
    return generate(tokens, 200)

#%%
state = torch.load('model.pth', map_location='cuda')
cTransformer.load_state_dict(state)
cTransformer.eval()
#%%

#text = input('> ')
cTransformer.eval()
output = Prompt("a\n")
print(output)

