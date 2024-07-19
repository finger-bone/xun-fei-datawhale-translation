#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
from torch import Tensor


# In[2]:


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int=1024):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-np.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, embedding_dim]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x  # [seq_len, batch_size, embedding_dim]


# In[3]:


class InputBlock(nn.Module):
    
    def __init__(self, embed_d, src_vocab, max_len=1024, dropout=0.1):
        super(InputBlock, self).__init__()
        self.embed = nn.Embedding(src_vocab, embed_d)
        self.pe = PositionalEncoding(embed_d, max_len)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.pe(x)
        return self.dropout(x)


# In[4]:


class AddAndNorm(nn.Module):
    
    def __init__(self, embed_d, dropout=0.1):
        super(AddAndNorm, self).__init__()
        self.norm = nn.LayerNorm(embed_d)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, y):
        return self.norm(x + self.dropout(y))


# In[5]:


class MultiHeadAttn(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttn, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1 / np.sqrt(self.d_k)
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, x, y, mask=None):
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(y))
        V = self.split_heads(self.W_v(y))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return self.dropout(output)


# In[6]:


class FF(nn.Module):
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super(FF, self).__init__()
        self.sq = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.sq(x))


# In[7]:


class EncBlock(nn.Module):
    
    def __init__(self, d: int, num_heads: int, hidden_dim: int, dropout: float):
        super(EncBlock, self).__init__()
        self.mha = MultiHeadAttn( d, num_heads, dropout)
        self.ff = FF(d, hidden_dim, dropout)
        self.add_norm1 = AddAndNorm(d, dropout)
        self.add_norm2 = AddAndNorm(d, dropout)
    
    def forward(self, x, y, mask=None):
        x = self.add_norm1(x, self.mha(x, y, mask))
        return self.add_norm2(x, self.ff(x))


# In[8]:


class DecBlock(nn.Module):
    
    def __init__(self, d: int=512, num_heads: int=8, hidden_dim: int=1024, dropout: float=0.1):
        super(DecBlock, self).__init__()
        self.mha = MultiHeadAttn(d, num_heads, dropout)
        self.add_and_norm1 = AddAndNorm(d, dropout)
        self.cross_mha = MultiHeadAttn(d, num_heads, dropout)
        self.add_and_norm2 = AddAndNorm(d, dropout)
        self.ff = FF(d, hidden_dim, dropout)
        self.add_and_norm3 = AddAndNorm(d, dropout)
    
    def forward(self, x, y, src_mask=None, trg_mask=None):
        x = self.add_and_norm1(x, self.mha(x, x, trg_mask))
        x = self.add_and_norm2(x, self.cross_mha(x, y, src_mask))
        x = self.add_and_norm3(x, self.ff(x))
        return x


# In[9]:


def generate_mask(src, tgt):
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
    seq_length = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=device), diagonal=1)).bool()
    tgt_mask = tgt_mask & nopeak_mask
    return src_mask, tgt_mask


# In[10]:


class Transformer(nn.Module):
    
    def __init__(self, src_vocab: int, tgt_vocab: int, d: int=512, num_heads: int=8, hidden_dim: int=2048, num_enc: int=6, num_dec: int=6, dropout: float=0.1):
        super(Transformer, self).__init__()
        self.src_embed = InputBlock(d, src_vocab)
        self.tgt_embed = InputBlock(d, tgt_vocab)
        self.encs = nn.ModuleList([
            EncBlock(d, num_heads, hidden_dim, dropout) for _ in range(num_enc)
        ])
        self.decs = nn.ModuleList([
            DecBlock(d, num_heads, hidden_dim, dropout) for _ in range(num_dec)
        ])
        self.fc = nn.Linear(d, tgt_vocab)
    
    def forward(self, src, trg):
        # src: (batch_size, src_len)
        # trg: (batch_size, trg_len)
        src_mask, trg_mask = generate_mask(src, trg)
        src = self.src_embed(src)
        trg = self.tgt_embed(trg)
        
        for enc in self.encs:
            src = enc(src, src, src_mask)
        for dec in self.decs:
            trg = dec(trg, src, src_mask, trg_mask)
        
        return self.fc(trg)


# In[11]:


from transformers import AutoTokenizer
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


# In[12]:


class MTTrainDataset(Dataset):
    
    def __init__(self, train_path, dic_path):
        self.terms = [
            {"en": l.split("\t")[0], "zh": l.split("\t")[1]} for l in open(dic_path).read().split("\n")[:-1]
        ]
        self.data = [
            {"en": l.split("\t")[0], "zh": l.split("\t")[1]} for l in filter(
                lambda x: len(x) < 512,
                open(train_path).read().split("\n")[:-1]
            )
        ]
        self.en_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", cache_dir="../../cache")
        self.ch_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese", cache_dir="../../cache")
        self.en_tokenizer.add_tokens([
            term["en"] for term in self.terms
        ])
        self.ch_tokenizer.add_tokens([
            term["zh"] for term in self.terms
        ])
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index) -> dict:
        return {
            "src": Tensor(self.en_tokenizer.encode(self.data[index]["en"])).to(device, dtype=torch.long), 
            "trg": Tensor(self.ch_tokenizer.encode(self.data[index]["zh"])).to(device, dtype=torch.long)
        }
    
    def get_raw(self, index):
        return self.data[index]


# In[13]:


train_data = MTTrainDataset("./data/train.txt", "./data/en-zh.dic")


# In[14]:


device = "cuda"


# In[15]:


model = Transformer(
    len(train_data.en_tokenizer), len(train_data.ch_tokenizer)
).to(device)


# In[16]:


model.train()
pass


# In[17]:


def collate_fn(batch):
    src = torch.nn.utils.rnn.pad_sequence([x["src"] for x in batch], batch_first=True, padding_value=0)
    trg = torch.nn.utils.rnn.pad_sequence([x["trg"] for x in batch], batch_first=True, padding_value=0)
    return src, trg


# In[18]:


# # set random values for the model
# for p in model.parameters():
#     if p.dim() > 1:
#         nn.init.xavier_uniform_(p)


# In[19]:


def train(
    epochs: int=10,
    steps: int | None=None,
    batch_size: int=4,
    logging_times: int=200,
    check_each_epoch_times: int=3,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, scheduler_steps, gamma=gemma)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    losses = []
    logging_steps = (steps if steps is not None else (len(train_data) // batch_size)) // logging_times
    check_steps = (steps if steps is not None else (len(train_data) // batch_size)) // check_each_epoch_times
    from tqdm import tqdm
    data_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)
    for epoch in tqdm(range(epochs)):
        for step, (src, trg) in tqdm(
            enumerate(data_loader), total=len(data_loader) if steps is None else steps, desc=f"Epoch: {epoch}"
        ):
            src = src.to(device)
            trg = trg.to(device)
            optimizer.zero_grad()
            output = model(src, trg[:, :-1])
            loss = criterion(output.reshape(-1, output.size(-1)), trg[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # scheduler.step()
            losses.append(loss.item())
            if steps is not None and step >= steps:
                break
            if step % logging_steps == 0:
                print(f"Epoch: {epoch}, Step: {step}, Loss: {losses[-1]}")
                print(f"avg_loss: {np.mean(losses[-logging_steps:])}")
                print(f"Input: {train_data.en_tokenizer.decode(src[0].tolist())}")
                print(f"Prediction: {train_data.ch_tokenizer.decode(output.argmax(-1)[0].tolist())}")
                print(f"Target: {train_data.ch_tokenizer.decode(trg[0].tolist())}")
                print("=" * 100)
            if step % check_steps == 0:
                torch.save(model.state_dict(), f"./m_{step}_{epoch}.pth")
    return losses


# In[20]:


losses = train(epochs=3, steps=None, logging_times=100, batch_size=32)


# In[ ]:


model.train()
pass


# In[ ]:

with open("submit.txt", "w") as f:
    lines = open("./data/test_en.txt").read().split("\n")[:-1]
    for line in lines:
        src = Tensor(train_data.en_tokenizer.encode(line)).to(device, dtype=torch.long).unsqueeze(0)
        output = Tensor(
            [train_data.ch_tokenizer.cls_token_id]
        ).to(device, dtype=torch.long).unsqueeze(0)
        while output.size(1) < 512 and output[0, -1] != train_data.ch_tokenizer.sep_token_id:
            output = torch.cat([output, model(src, output)[:, -1].argmax(-1).unsqueeze(1)], dim=1)
        f.write(train_data.ch_tokenizer.decode(output[0].tolist(), skip_special_tokens=True) + "\n")


