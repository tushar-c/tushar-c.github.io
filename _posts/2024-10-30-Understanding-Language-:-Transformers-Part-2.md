
In this post, we will continue from where we left off in the last post. We will march on towards our ultimate goal of implementing a Transformer from scratch. The full implementation of the Transformer can be found [here:](https://gist.github.com/tushar-c/d0c6b51822f1daf067b51cb68788acd7)

We will use the [PyTorch Framework](https://pytorch.org/) to create our implementation of the Transformer. Also, as mentioned previously, our implementation is based on the **Attention is All you Need** Paper by **Vaswani et. al.** found [here](https://arxiv.org/abs/1706.03762).

## Introduction

In this post, we will focus on implementing all the building blocks of the Transformer Architecture. Since the Encoder and the Decoder (explained later), are the core parts of the Transformer, the parts that we will focus on in this post also serve as the building blocks of the Encoder and the Decoder. Let's begin!

We saw in the previous post that we must process our input sentence by converting words into one-hot vectors that we later process to neural networks so that we can make the network learn from the inputs and establish an input-output mapping through the backpropagation algorithm. 

However, as we can see, only having a one hot vector in the dimension of the vocabulary limits our ability to be _expressive_ when it comes to learning. 

We want to use not just one position of the vector, as is the case for one-hot vectors, but we want to use all the positions in the vector. Moreover, we don't want to use just a single number for representing a whole word. Instead, we would like to use all real-valued numbers for our purposes. 

As is evident, we would like to use the whole vector and have real-valued values across each and every position. 

## Enter: The Embedding Matrix

For this very exact purpose, we have the Embedding Matrix. An Embedding Matrix does as its name suggests. It takes a one-hot vector in a certain dimension _'N'_ and then projects that vector into another dimension _'M'_ having real-valued numbers across all of the positions of the vector. 

This is enabled by the Embedding Matrix. In technical terms, an Embedding Matrix is a Linear Transformation that takes an _'N-dimensional'_ vector and projects it into an _'M-dimensional'_ vector.

Therefore, for the case mentioned above, the Embedding Matrix is an N x M Matrix for our purposes. For the purposes of the Transformer, we tokenize each sentence (a collection of words) and then apply the Input Embeddings to the said tokenized input.

## Building Code Blocks

We can see how this is done in code now. We create a file called `transformer.py`, but you are free to give it any name you wish. 

Before we start, we need to do the following imports:

```
import torch
import torch.nn as nn
import numpy as np
```

After this is done, we can now set up some parameters and implement the word and sentence tokenization architecture. We will first create some parameters as shown below. Some of these won't be obvious right now and they will be explained in detail further in this post.

```
# we have a total vocabulary of 747 words
vocab_size = 747

# we choose 12 words per sentence
context_length = 12

# we chose 9 sentences at a time
batch_size = 9

# D_MODEL here is basically `H` number of attention heads, each producing outputs in the `D_V` dimension, concatenated
D_MODEL = 512

# we define the number of attention heads
NUM_HEADS = 8

# we also define the `D_K` and `D_V` parameters for the encoder and decoder
D_K = int(D_MODEL / NUM_HEADS)
D_V = int(D_MODEL / NUM_HEADS)

# set up the training output parameters for the decoder

# vocabulary size for the outputs
output_vocab_size = 879

print('Transformer Model Parameters Loaded...')
print(f'dk = {D_K}; dv = {D_V}; dmodel = {D_MODEL}')
```

The `vocab_size` is a variable that we have created regarding the vocabulary size of our input language. The `context_length` is a variable that describes how many words from the input sequence will we consider at once when producing an output using the Transformer. The `batch_size` variable decides how many sentences will we process at once.

Do not worry about the `D_MODEL`, `NUM_HEADS`, `D_K`, `D_V` variables right now! They are all variables that we will explain later in this post.

In the end, we just print the values of the variables as a quick logging exercise.

## Embedding the Tokenizations

Now we begin the process of tokenizing our inputs and applying the embeddings to the inputs so that we may feed these inputs to the Transformer. We start by one-hot encoding the words.

```
def create_one_hot_vector(ix):
    one_hot_vector = torch.zeros((1, vocab_size))
    # the vector has the value of `1` at the indicated index and is `0` everywhere else
    one_hot_vector[0][ix] = 1
    return one_hot_vector
```

As is evident, this piece of code takes an `ix` value (an index value) and then creates a vector of `vocab_size` dimension, of all zeros and sets the `ix` position to `1`. Remember that we have zero-indexing so that the first index is `1`.
We then return the vector.












