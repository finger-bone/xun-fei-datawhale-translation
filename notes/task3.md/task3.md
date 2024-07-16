# Dawn of the Transformer

## Introduction

The previous part of the series presents traditional NLP models. This part will concern itself with the best NLP model up to date, completely new architecture that will be built from scratch.

This notebook will introduce the transformer architecture in the order of input to output.

Transformer architecture is also a multi-encoder-multi-decoder architecture. Some models also only contains the encoder part, which is used for tasks like text classification, whereas some models only contain the decoder part, which is used for tasks like text generation. For machine translation, both encoder and decoder should be used since the text generation and text understanding are both required.

## The Transformer Architecture for Machine Translation

### Encoder

#### Input Embedding and Positional Encoding

Embedding and tokenization have already been introduced in previous parts.

Except for the normal tokenization and embedding, another important part of the input is the positional encoding in the transformer architecture.

The necessity of positional encoding is justified by the fact that the transformer architecture does not have any recurrence or convolution, in other words, it doesn't process the input token-by-token, and thus it fails to capture the position of the tokens in the input sequence.

To deal with the problem, instead of sending in only the embeddings of the tokens, the positional encoding is added to the embeddings. The positional encoding is a vector that is added to the embeddings of the tokens, and it is calculated by the following formula,

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i / d_{model}})$$

$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i / d_{model}})$$

where $pos$ is the position of the token in the input sequence, $i$ is the index of the dimension of the positional encoding, and $d_{model}$ is the dimension of the model.