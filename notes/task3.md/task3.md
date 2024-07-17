# Dawn of the Transformer

## Introduction

The previous part of the series presents traditional NLP models. This part will concern itself with the best NLP model up to date, completely new architecture that will be built from scratch.

This notebook will introduce the transformer architecture in the order of input to output.

Transformer architecture is also a multi-encoder-multi-decoder architecture. Some models also only contains the encoder part, which is used for tasks like text classification, whereas some models only contain the decoder part, which is used for tasks like text generation. For machine translation, both encoder and decoder should be used since the text generation and text understanding are both required.

## The Transformer Architecture for Machine Translation

### General Structure

This part introduces the transformer structure by breaking it down into several parts.

The general structure is as follows,

![Transformers Architecture](image.png)

Which can be expressed as,

```
enc_out = enc_in |> input_block |> [multi_head_self_attention |> add_and_norm |> feed_forward |> add_and_norm] * N

dec_out = [(dec_out |> multi_head_self_attention |> add_and_norm |> feed_forward |> add_and_norm, enc_out) |> multi_head_cross_attention |> add_and_norm |> feed_forward |> add_and_norm] * N |> un_embedding_block
```

where `N` is the number of layers in the transformer architecture, `|>` is pipe, and `[...]` is the list of functions that are applied in order.

### Input Block

Embedding and tokenization have already been introduced in previous parts.

Except for the normal tokenization and embedding, another important part of the input is the positional encoding in the transformer architecture.

The necessity of positional encoding is justified by the fact that the transformer architecture does not have any recurrence or convolution, in other words, it doesn't process the input token-by-token, and thus it fails to capture the position of the tokens in the input sequence.

To deal with the problem, instead of sending in only the embeddings of the tokens, the positional encoding is added to the embeddings. The positional encoding is a vector that is added to the embeddings of the tokens, and it is calculated by the following formula,

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i / d_{model}})$$

$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i / d_{model}})$$

where $pos$ is the position of the token in the input sequence, $i$ is the index of the dimension of the positional encoding, and $d_{model}$ is the dimension of the model, which equals the dimension of the embeddings.

This equal may seem arbitrary, but it is chosen to make the positional encoding have a smooth curve, which means that the positional encoding will have a similar value for similar positions.

In addition, the positional coding using the sine and cosine functions is chosen because the model can learn to attend to relative positions, since the sine of the sum of two angles can be expressed as a function of the sines and cosines of the angles, and so is the cosine.

### Multi-head Self Attention

#### Cross Attention

If an attention layer requires to pay attention to a sequence based on another sequence, it is called cross attention. For example, the decoder in the machine translation task should pay attention to the encoder output in order to process the output of previous decoder layers.

The cross attention is calculated by the following formula,

$$
Q = YW^Q \\
K = XW^K \\
V = XW^V \\
$$

So the cross attention can be calculated the same as the self-attention, but the queries are based on the desired output shape.

#### The Mask Technique

There is another trick that improves the self-attention mechanism.

The `mask` is used to prevent the model from attending to the future tokens in the input sequence. The `mask` is a matrix that is added to the attention scores, and it is calculated by the following formula,

$$
\text{mask}_{ij} = \begin{cases} -\infty & \text{if } j > i \\ 0 & \text{otherwise} \end{cases}
$$

where $i$ is the row index and $j$ is the column index of the matrix.

So the `mask` is added to the attention scores before the soft-max function is applied to the attention scores, and the model won't attend to the future tokens in the input sequence.

Furthermore, for special tokens like the padding token, the `mask` is also used to prevent the model from attending to the padding tokens, which can be done by setting the `mask` value to $-\infty$ for the padding tokens.

So to conclude, the mask should be,

$$
\text{mask}_{ij} = \begin{cases} -\infty & \text{if } j > i \text{ or } \text{input}[i] \text{ is } \text{padding token} \\ 0 & \text{otherwise} \end{cases}
$$

And `mask` should be applied to the attention scores before the soft-max function is applied to the attention scores.

### Add and Norm

The add and norm operation is a layer that is added after every sub-layer in the transformer architecture. The add and norm operation is defined as,

$$\text{AddAndLayerNorm}(x)=\text{LayerNorm}(x + \text{SubLayer}(x))$$

where $x$ is the input to the sub-layer, and $\text{SubLayer}(x)$ is the output of the sub-layer.

This operation is used to prevent the model from exploding or vanishing gradients, and it is also used to stabilize the training process.

Residual connections is beneficial for gradient flow because it allows the gradients to flow through the network without vanishing or exploding.

This step will be applied after every layer in the transformer architecture. So it will not be repeated in the following sections.

### Feed Forward

The feed forward layer is a simple layer that is used to transform the input to a higher dimension. The feed forward layer is defined as,

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

where $x$ is the input to the feed forward layer, $W_1$ and $W_2$ are the weights of the feed forward layer, and $b_1$ and $b_2$ are the biases of the feed forward layer.

The layer is basically just a traditional multi-layer linear neural network with a ReLU activation function.

### Un-Embedding Block

Un-embedding block is the same as previous parts. It just converts from embedding back to vocabulary vector, and if needed, further into token ids.

## Implementation

### Add and Norm

```python
class AddAndNorm(nn.Module):
    def __init__(self, dim, dropout):
        super(AddAndNorm, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, y):
        return self.norm(x + self.dropout(y))
```

### Feed Forward

```python
class FeedForward(nn.Module):
    def __init__(self, size, hidden_size, dropout):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
```

### Multi-head Self Attention

```python
class SelfAttention(nn.Module):

    def __init__(self, size, dropout)

### Input Block

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
    
    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()
```

```python

```