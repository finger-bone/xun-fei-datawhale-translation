# The Night Before the Transformer

## Introduction

The second task aims to improve the performance of the original encoder-rnn-decoder model without introducing the transformer architecture, albeit it has been proven, currently, the best architecture for machine translation tasks.

## A Recapitulation of the Previous Model

In task one, there was a encoder-decoder model with rnn networks. However, it behaved poorly due to, mainly, the context compression. The encoder compresses the context into a fixed-length vector, which is not enough to store all the information. It may work for short sentences, but it fails for longer ones.

However, that model got the greater part of the job done. It could translate simple sentences, but it failed for more complex ones. The model was trained on a small dataset, which is not enough to learn the complex patterns of the language. Now, we just need to improve the encoder and decoders.

## Attention Mechanism

Attention mechanism, to put simply, is to weight the importance of different parts of the input sequence. It is a mechanism that allows the model to focus on different parts of the input sequence.

A obvious way to improve the encoder-decoder model is to add an extra attention layer in between the encoder and decoder. The attention layer will help the decoder to focus on different parts of the input sequence.

An example is as below,

```python
class Attention(nn.Module):
    def __init__(self, hidden_dim, d):
        super(Attention, self).__init__()
        self.w = nn.Linear(hidden_dim * 2, d)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.activation = nn.Tanh()
    
    def forward(self, x, h):
        h = h.permute(1, 0, 2)
        # h is [batch, num_layers == 1, hidden_dim]
        # x is [batch, len, hidden_dim]
        h = h.expand(-1, x.size(1), -1)
        # [batch, len, hidden_dim * 2] -> [batch, len, d]
        w = self.w(th.cat((x, h), dim=-1))
        # [batch, len, d] -> [batch, len, 1] -> [batch, len]
        attn = self.v(w)
        attn = attn.squeeze(-1)
        return th.softmax(attn, dim=-1)
```

```python
class Decoder(nn.Module):
    
    def __init__(self, zh_vocab_size, embed_dim=256, hidden_dim=1024, drop_out_rate=0.1) -> None:
        super().__init__()
        # -> [batch, len]
        self.attn = Attention(hidden_dim, hidden_dim)
        # [batch, len == 1] -> [batch, len == 1, embed_dim]
        self.embed = nn.Embedding(zh_vocab_size, embed_dim)
        # [len == 1, batch, embed_dim + hidden_dim] -> [len == 1, batch, hidden_dim], [n_layers, batch, hidden_dim]
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim)
        # [batch, hidden_dim] -> [batch, zh_vocab_size]
        self.fc = nn.Linear(hidden_dim, zh_vocab_size)
        self.dropout = nn.Dropout(drop_out_rate)
        
    def forward(self, x, h, enc_out):
        # enc_out: [batch, len, hidden_dim]
        # x is [batch, len == 1]
        # h is [n_layers == 1, batch, hidden_dim]
        
        attn = self.attn(enc_out, h)
        # [batch, 1, hidden_dim] = [batch, 1, len] * [batch, len, hidden_dim]
        v = th.bmm(attn.unsqueeze(1), enc_out)
        # v: [batch, 1, hidden_dim]
        
        x = self.embed(x)
        # x: [batch, len == 1, embed_dim]
        x = self.dropout(x)
        rx = th.cat((v, x), dim=-1)
        # rx: [batch, len == 1, embed_dim + hidden_dim]
        rx = rx.permute(1, 0, 2)
        out_x, h = self.gru(rx, h)
        out_x = out_x.permute(1, 0, 2)
        out_x = self.fc(out_x.squeeze(1))
        return out_x, h
```

## Self Attention Mechanism

### Query, Key, and Value

Self attention is a mechanism that allows the model to focus on different parts of the input sequence. It takes three inputs, query, key, and value, and returns a weighted sum of the values. The query, key, and value are linear transformations of the input sequence.

There are three important concepts in the attention mechanism: query, key, and value.

- Query, the subjective attention vector. It is used to signify how subjectively important part of the value is to the current part of the sequence. By subjectively, it means that the model decides how important the part of the value is to the current part of the sequence, based on its previous knowledge.

- Key, the objective attention vector. It is used to signify how objectively important part of the value is to the current part of the sequence. By objectively, it means that part of the value is inherently more important to the current part of the sequence. For example, link verbs are less important than nouns.

- Value, the content vector. It is the part of the sequence that the model should focus on.

### Self Attention Math

In the self attention layer, firstly, a tensor of shape $(len, embed)$ will be transformed into three tensors with the shape $(len, d)$, where $d$ is the dimension, can be arbitrary, usually equal to the dimension of the embedding vector, so that the input and output have the same shape.

Then the math is as follows,

$$
\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d}})V
$$

where $d$ is the dimension of the key vector.

Or for clarity, using the abstract indices,

$$
\text{Attention}(Q_i^j, K_i^j, V_i^j) = \text{softmax}(\frac{Q_i^jK_j^k}{\sqrt{d}})V_k^l
$$

Noticing that query and key are actually fungible, our previous definition of query and key is just for the sake of explanation. How it really works is still shrouded in mystery.

The divisor $\sqrt{d}$ is used to prevent the dot product from being too large, which may result in the soft-max function returning 0 or 1. It is purely a practical trick.

### Implementation

A simple implementation of the self attention mechanism is as follows,

```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed, d):
        super(SelfAttention, self).__init__()
        self.Q = nn.Linear(embed, d)
        self.K = nn.Linear(embed, d)
        self.V = nn.Linear(embed, d)
        
    
    def forward(self, x):
        # x is [batch, len, embed]
        # Q, K, V are [batch, len, d]
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        # Q, K, V are [batch, len, d]
        # QK^T is [batch, len, len]
        # QK^T / sqrt(d) is [batch, len, len]
        # softmax(QK^T / sqrt(d)) is [batch, len, len]
        # softmax(QK^T / sqrt(d))V is [batch, len, d]
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        return out
```

## Integrate Self Attention into the RNN

