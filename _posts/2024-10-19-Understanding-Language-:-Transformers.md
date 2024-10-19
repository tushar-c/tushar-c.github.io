## Introduction

After some time away, we are back! This time, we are going to understand the idea of modelling human language through the framework of Machine Learning. More specifically, the most widely known and the most commercialized method of application for this task is the Large Language Model. This is basically a term for a machine learning model having many parameters that attempts to model human (or natural) language.

Presently, the underlying model that is used to implement this is the **"Transformer"**. The whole model has been described and explained in its entirety in the paper title **"Attention Is All You Need" by Vaswani et. al**. It is this model that we are going to understand and implement in this post.

## Foundations

Before we try to approach the model and its parameters and dive into the mathematics of the model, we need to understand what it is exactly that we are doing. We need to know the problem, we need to know why do we aim to solve the problem, we also need to know the solution and we also need to understand the solution.

In the most eagle-eyed sense, we are trying to establish a relation between two sets of words. These two sets of words are very large, differ in size, and sometimes they may even belong to different languages. We aim to give the computer a set of words as input and we are then asking different questions such as:

1. Given this set of words, what is the most probable set of words that come next in the sequence?

2. Given this set of words (in one language), what is the most probable set of words in another given 		  language?

These are just two example questions that we may ask the computer. We can ask even more questions! However, for now, this will have serve as an exhaustive list of the curiousity of the human mind when it comes to asking natural language questions to computers. As you may have understood, given the current sequence of words, the computer is going to output **probabilities** of the next words in the sequence.

Now, given that computers only understand binary and the best method we have of communicating directly with them is through programming languages, how do we feed everyday language to a computer and expect it to give us everyday language as output that makes sense?

## Understanding the Transformer
