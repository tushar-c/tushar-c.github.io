
In this post, we will continue from where we left off in the last post. We will march on towards our ultimate goal of implementing a Transformer from scratch. The full implementation of the Transformer can be found [here:](https://gist.github.com/tushar-c/d0c6b51822f1daf067b51cb68788acd7)

We will use the [PyTorch Framework](https://pytorch.org/) to create our implementation of the Transformer. Also, as mentioned previously, our implementation is based on the **Attention is All you Need** Paper by **Vaswani et. al.** found [here](https://arxiv.org/abs/1706.03762).

## Introduction

In this post, we will focus on implementing all the building blocks of the Transformer Architecture. Since the Encoder and the Decoder (explained later), are the core parts of the Transformer, the parts that we will focus on in this post also serve as the building blocks of the Encoder and the Decoder. Let's begin!

We saw in the previous post that we must process our input sentence by converting words into one-hot vectors that we later process to neural networks so that we can make the network learn from the inputs and establish an input-output mapping through the backpropagation algorithm. 

However, as we can see, only having a one hot vector in the dimension of the vocabulary limits our ability to be _expressive_ when it comes to learning. 

We want to use not just one position of the vector, as is the case for one-hot vectors, but we want to use all the positions in the vector. Moreover, we don't want to use just a single number for representing a whole word. Instead, we would like to use all real-valued numbers for our purposes. 

As is evident, we would like to use the whole vector and have real-valued values across each and every position. 

## Enter: The Embedding Matrix

For this very exact purpose, we have the Embedding Matrix. An Embedding Matrix does as its name suggests. It takes a one-hot vector in a certain dimension _'N'_ and then projects that vector into another dimension _'M'_ having real-valued numbers across all of the positions of the vector. This is enabled by the Embedding Matrix. In technical terms, an Embedding Matrix is a Linear Transformation that takes an _'N-dimensional'_ vector and projects it into an _'M-dimensional'_ vector.

Therefore, for the case mentioned above, the Embedding Matrix is an N x M Matrix for our purposes.




