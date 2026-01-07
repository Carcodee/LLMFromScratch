#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import ByteLevelBPETokenizer
import os
import re
import math
import matplotlib.pyplot as plt
from datetime import datetime

# %%

df = pd.read_csv('./text_data/data.csv')
df = df.drop(['Dataline','PlayerLinenumber','Play','ActSceneLine'], axis=1)
df[['Player', 'PlayerLine']] += '\n'
str_list = df.values.flatten().tolist()
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

#%%
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
context_lenght = 32
pos_emb_lenght = 512
batch_count = 256
lr = 1e-3
vocab_size = tokenizer.get_vocab_size()
emb_dim= 512
head_count = 8 
feed_forward = emb_dim * 4
temperature = 1

#%%
def batchify_data(src_data):
    src_usable = src_data[: src_data.shape[0] - (src_data.shape[0] % context_lenght) + 1]
    inputs =src_usable[:-1].view(-1, context_lenght)
    targets =src_usable[1:].view(-1, context_lenght)
    return inputs, targets


def get_batches(batches_count, input_batches, target_batches):
    all_starts_inputs= torch.randint(0, input_batches.shape[0], size=(batches_count, ), dtype=torch.long).to(device=device)

    inputs = input_batches[all_starts_inputs]
    targets = target_batches[all_starts_inputs]
    return inputs, targets

#%%
class DataLoader():
    def __init__(self):
        self.offset = 0;
        self.curr_str = ''
        self.batch_idx = 0;
        pass

    def SetInfo(self, doc_size, path, train_size, tokenizer):
        self.curr_path = path
        self.total_size = os.path.getsize(self.curr_path)
        self.doc_size = self.total_size if doc_size == -1 else doc_size
        self.total_documents = int(self.total_size/self.doc_size) 
        self.train_size = train_size
        self.tokenizer = tokenizer

    def Next_doc(self):
        with open(self.curr_path, 'rb') as f:
            f.seek(self.offset)
            chunk = f.read(self.doc_size)
            self.curr_str = chunk.decode('utf-8')
        self.doc_data = self.tokenizer.encode(self.curr_str)
        self.offset = self.offset + self.doc_size
        finish_file = False
        if self.offset >= self.total_size:
            self.offset = 0
            finish_file = True
        return finish_file

    def BuildData(self):
        assert(self.curr_path != '')
        self.Next_doc()
        self.src = torch.tensor(self.doc_data.ids, device=device)
        batched_inputs, batched_targets = batchify_data(self.src) 
        shuffled_idx = torch.randperm(batched_inputs.shape[0]).to(device=device)
        batched_inputs = batched_inputs[shuffled_idx]
        batched_targets = batched_targets[shuffled_idx]

        self.batched_inputs_train = batched_inputs[:int(self.train_size * batched_inputs.shape[0])]
        self.batched_targets_train = batched_targets[:int(self.train_size * batched_inputs.shape[0])]

        self.batched_inputs_val = batched_inputs[int(self.train_size * batched_inputs.shape[0]):]
        self.batched_targets_val = batched_targets[int(self.train_size * batched_inputs.shape[0]):]

    def Next(self, batch_count):
        final_batch_size = self.batch_idx + batch_count - self.batch_idx 
        x_out = self.batched_inputs_train[self.batch_idx: self.batch_idx + final_batch_size, :]
        y_out = self.batched_targets_train[self.batch_idx: self.batch_idx + final_batch_size, :]
        self.batch_idx += final_batch_size
        finish_doc = False
        if self.batch_idx >= self.batched_inputs_train.shape[0]:
            self.batch_idx = 0
            self.BuildData()
            finish_doc = True 
        return x_out, y_out, finish_doc
    def GetVal(self, batches_count):
        return get_batches(batches_count, self.batched_inputs_val, self.batched_targets_val) 



# %%

dl = DataLoader()
dl.SetInfo(-1, './text_data/final_training_data.txt', 0.9, tokenizer)
dl.BuildData()
print(dl.batched_inputs_train.shape)
print(dl.batched_inputs_val.shape)

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

inputs, targets, _ = dl.Next(32)
print(inputs.shape)
print(targets.shape)


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
        wei = F.dropout(wei, p=0.1, training=self.training)
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
        out = F.dropout(self.proj(out), p=0.1, training=self.training)
        return out


class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, input_size),
            nn.Dropout(0.1),
        )
    def forward(self, x):
        return self.model(x)

class Block(nn.Module):
    def __init__(self, head_count, emb_dim):
        super().__init__()
        self.lyn1 = nn.LayerNorm(emb_dim)
        self.mhatt = MultiHeadAttention(head_count, emb_dim)
        self.lyn2 = nn.LayerNorm(emb_dim)
        self.ff = FeedForward(emb_dim, feed_forward)
        self.dropout = nn.Dropout(0.1)
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
        )
        self.register_buffer(
            "pos_ids",
            torch.arange(pos_emb_lenght)
        )
        
        self.ln_f = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)
    def forward(self, x_in, target = None):
        B, T = x_in.shape
        emb_tok = self.tok_emb(x_in).to(device)
        pos = self.pos_ids[:T].unsqueeze(0)
        emb_pos = self.pos_emb(pos)
        x = emb_pos + emb_tok
        x = F.dropout(x, p=0.1, training=self.training)
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
optimizer = torch.optim.AdamW(cTransformer.parameters(), lr=lr, weight_decay=0.01)
param_counter = 0

#for p in cTransformer.parameters():
#    p+= p.numel()

#print(param_counter)

#cTransformer = torch.compile(cTransformer, backend='eager')

#%%
#cTransformer.load_state_dict(torch.load('checkpoint.pth'))

#%%
def Train(epochs, dl, print_ts = False):

    losses = []
    for n in range(epochs):
        steps_per_epoch = int(dl.batched_inputs_train.shape[0] / batch_count) - int(dl.batch_idx/batch_count)
        print(f'Steps per epoch: {steps_per_epoch}')
        for step in range(steps_per_epoch):
            x_train, y_train, finish_epoch = dl.Next(batch_count)
            optimizer.zero_grad(set_to_none = True)
            y, loss = cTransformer(x_train, y_train)
            loss.backward()
            optimizer.step()
            if n % 20 == 0 or step == 20:
                with torch.no_grad():
                    if print_ts:
                        time_start = datetime.now().microsecond
                    input_test, target_test = dl.GetVal(32)
                    _, loss_test = cTransformer(input_test, target_test)
                    losses.append((loss, loss_test))
                    output = f'epoch: {n}, step: {step}/{steps_per_epoch} train: {loss:.4f}, test: {loss_test:.4f}'
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
losses = Train(2, dl)
epoch_counter+= 2


#%%
losses_post = Train(2, dl, True)
epoch_counter+= 2 

#%%
torch.save({'model': cTransformer.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch_counter},
            'checkpoint.pth')
#%%
losses_train = []
losses_val = []
for loss in losses:
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

