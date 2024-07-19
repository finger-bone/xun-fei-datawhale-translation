# The Night Before the Transformer

## Introduction

The second task aims to improve the performance of the original encoder-rnn-decoder model without introducing the transformer architecture, albeit it has been proven, currently, the best architecture for machine translation tasks.

## A Recapitulation of the Previous Model

In task one, there was a encoder-decoder model with rnn networks. However, it behaved poorly due to, mainly, the context compression. The encoder compresses the context into a fixed-length vector, which is not enough to store all the information. It may work for short sentences, but it fails for longer ones.

However, that model got the greater part of the job done. The model was trained on a small dataset, which is not enough to learn the complex patterns of the language. Now, we just need to improve the encoder and decoders.

## Linear Attention Mechanism

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

In the code above, the `attn` assumes the role of the attention, which is a tensor of shape $(batch, len)$, where $len$ is the length of the input sequence. So, each element of the tensor refers to the importance of the corresponding part of the input sequence.

Using that, we can pay attention to the output vector of the encoder so that we can focus on different parts of the input sequence.

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
        # [batch, hidden_dim * 2 + embed_dim] -> [batch, zh_vocab_size]
        self.fc = nn.Linear(hidden_dim * 2 + embed_dim, zh_vocab_size)
        self.dropout = nn.Dropout(drop_out_rate)
        self.activation = nn.Tanh()
        
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
        rx = self.activation(rx)
        # rx: [batch, len == 1, embed_dim + hidden_dim]
        rx = rx.permute(1, 0, 2)
        out_x, h = self.gru(rx, h)
        out_x = out_x.permute(1, 0, 2)
        # out_x: [batch, len == 1, hidden_dim]
        out_x = out_x.squeeze(1)
        v = v.squeeze(1)
        fc_in = th.cat((out_x, v, x.squeeze(1)), dim=-1)
        
        out_x = self.fc(fc_in)
        return out_x, h
```

The result is still horrible. Yet, compared to the previous model, the model can get the length of the sentence almost correct, and it shows some sign of understanding the sentence. After fully training on the provided dataset for one epoch, a much, much better score is achieved.

Nevertheless, the model is still scarcely usable.

## Dot-product Attention Mechanism

### Query, Key, and Value

Dot-product attention is a mechanism that allows the model to focus on different parts of the input sequence. It takes three inputs, query, key, and value, and returns a weighted sum of the values. The query, key, and value are linear transformations of the input sequence.

There are three important concepts in the attention mechanism: query, key, and value.

- Query, the subjective attention vector. It is used to signify how subjectively important part of the value is to the current part of the sequence. By subjectively, it means that the model decides how important the part of the value is to the current part of the sequence, based on its previous knowledge.

- Key, the objective attention vector. It is used to signify how objectively important part of the value is to the current part of the sequence. By objectively, it means that part of the value is inherently more important to the current part of the sequence. For example, link verbs are less important than nouns.

- Value, the content vector. It is the part of the sequence that the model should focus on.

When the query and key derive from the same input, it is called self attention. When the query and key derive from different inputs, it is called cross attention.

### Dot-product Attention Math

In the attention layer, firstly, a tensor of shape $(len, embed)$ will be transformed into three tensors with the shape $(len, d)$, where $d$ is the dimension, can be arbitrary, usually equal to the dimension of the embedding vector, so that the input and output have the same shape.

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

A simple implementation of the dor-product attention mechanism is as follows,

```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed, d):
        super(SelfAttention, self).__init__()
        self.Q = nn.Linear(embed, d)
        self.K = nn.Linear(embed, d)
        self.V = nn.Linear(embed, d)
        self.d = d
    
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
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.d ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        return out
```

## Multi-Head Attention

The attention mechanism can be improved by using multiple heads. The multi-head self attention mechanism is a mechanism that allows the model to focus on different parts of the input sequence.

To put it more simply, the multi-head mechanism is to use multiple self attention layers in parallel, after splitting the embedding dimension into small chunks, and then concatenate the results. The multi-head mechanism can help the model to focus on different parts of the input sequence.

The math is as follows,

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{Head}_1, \text{Head}_2, \ldots, \text{Head}_n)W^O
$$

where $\text{Head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$ is a sub-attention, and $W^O$ is a linear transformation.

An simple implementation is as follows, which is a cross attention because the query and key are different.

```python
class MultiHeadAttn(nn.Module):
    
    def __init__(
        self, 
        dim: int, 
        heads: int, 
        dropout: float
    ):
        super(MultiHeadAttn, self).__init__()
        
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        
        assert self.head_dim * heads == dim, "dim must be divisible by heads"
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        
        self.fc_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor, y: Tensor, mask: Tensor | None=None) -> Tensor:
        batch_size = x.shape[0]
        
        # Linear projections
        Q = self.q(y)
        K = self.k(x)
        V = self.v(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        energy = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # Adjust mask shape for broadcasting
            energy = energy.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        out = torch.matmul(attention, V)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        
        # Final linear layer
        out = self.fc_out(out)
        
        return out
```

However, this part will not use the multi-head self attention mechanism. We will use them in the transformer architecture.

## Other Improvements

Without tweaking the model too much, here are some other improvements that can be made to the model.

- Use LSTM instead of GRU. LSTM is more powerful than GRU, and it can learn more complex patterns in the data.
- Increase the `n_layers` of the encoder and decoder. Increasing the number of layers will help the model to learn more complex patterns in the data.
- Use multi-head self attention in the encoder and decoder.
- Use more dropout since it is obvious that the model is over-fitting, for continuously generating the same words.
- Adding extra residual connections in the model.
- Performing layer normalization after each layer.
- Increase the size of `hidden_dim` and `d`.

During the training, there was also a `nan` loss problem caused by the `nan` in gradient. When encountering such problem, enforcing gradient clipping or simply decrease the learning rate can solve the problem.

Utilizing the previous mentioned techniques, a better result was achieved.

There are also other things to note.

- Make sure to increase `dropout` to a large value so that the model does not over-fit.
- If training on a multi-GPU environment, the model must be wrapped in `nn.DataParallel` to be able to run on multiple GPUs.
- `pytorch` has a sort of feature-like bug, that is, when creating multiple tensor with different shapes, it will create cache for each shape, which will consume a lot of memory. To solve this problem, use `torch.cuda.empty_cache()` to clear the cache, which may work. A better way to get around would be to pad the tensor to the same length or the multiple of the same length. I didn't know that before so it's not in the code, but the memory was substantially higher than it should be, and the training was very slow.
- enable truncating, or else some very long outliers will cause out of memory error.
