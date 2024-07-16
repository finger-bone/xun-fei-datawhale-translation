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

The encoder part consists of an input block to process the input, and several encoder blocks to process the input sequentially.

The decoder part consists of several pre-encoder decoder blocks to process the previously generated tokens, a next-to-encoder block, and several post-encoder decoder blocks to process the input sequentially, and eventually, a un-embedding block to generate the output.

Except for the input block and the un-embedding block, the other blocks are of the same structure, with a multi-head self-attention layer, a feed forward layer, and add and norm layers after each of them. The difference is only in the shape of the input and output of the blocks.

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

### Encoder Block

#### Multi-head Self Attention

The self-attention here has been previous introduced in the part of the series about attention mechanisms. However, there is another trick that improves the self-attention mechanism.

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

#### Add and Norm

The add and norm operation is a layer that is added after every sub-layer in the transformer architecture. The add and norm operation is defined as,

$$\text{AddAndLayerNorm}(x)=\text{LayerNorm}(x + \text{SubLayer}(x))$$

where $x$ is the input to the sub-layer, and $\text{SubLayer}(x)$ is the output of the sub-layer.

This operation is used to prevent the model from exploding or vanishing gradients, and it is also used to stabilize the training process.

Residual connections is beneficial for gradient flow because it allows the gradients to flow through the network without vanishing or exploding.

This step will be applied after every layer in the transformer architecture. So it will not be repeated in the following sections.

#### Feed Forward

The feed forward layer is a simple layer that is used to transform the input to a higher dimension. The feed forward layer is defined as,

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

where $x$ is the input to the feed forward layer, $W_1$ and $W_2$ are the weights of the feed forward layer, and $b_1$ and $b_2$ are the biases of the feed forward layer.

The layer is basically just a traditional multi-layer linear neural network with a ReLU activation function.

### Decoder Blocks

#### Pre-Encoder Decoder Block

The pre-encoder decoder block is a block that is used to process the input before the encoder and decoder blocks. For machine translation, the this block processes the previously generated tokens.

#### Next-to-Encoder Block

This block takes both the output of the pre-encoder decoder blocks and the output of the encoder blocks as input, pass them both through a multi-head self-attention layer, and then pass the output through a feed forward layer.

#### Post-Encoder Decoder Block

This block is used to process the input after the encoder and decoder blocks. For machine translation. This just takes the output of the previous block, and pass it through a multi-head self-attention layer and a feed forward layer.

### Un-Embedding Block

The un-embedding block is the same as the un-embedding from the previous parts of the series.

## Implementation

### Input Block

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register buffer ensures 'pe' is not considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to input tensor
        x = x + self.pe[:x.size(0), :]
        return x
```