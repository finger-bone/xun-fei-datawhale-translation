# Dawn of the Transformer

## Introduction

The previous part of the series presents traditional NLP models. This part will concern itself with the best NLP model up to date, completely new architecture that will be built from scratch.

This notebook will introduce the transformer architecture in the order of input to output.

Transformer architecture is also a multi-encoder-multi-decoder architecture. Some models also only contains the encoder part, which is used for tasks like text classification, whereas some models only contain the decoder part, which is used for tasks like text generation. For machine translation, both encoder and decoder should be used since the text generation and text understanding are both required.

## The Transformer Architecture for Machine Translation

### Encoder

#### Input Embedding and Positional Encoding

Except for the normal tokenization and embedding, another important part of the input is the positional encoding in the transformer architecture. The reason for that is that the transformer architecture does not have any recurrence or convolution, so the model does not know the order of the words in the sentence. The positional encoding is added to the input embeddings to give the model information about the position of the words in the sentence.
