## Introduction

After some time away, we are back! This time, we are going to understand the idea of modelling human language through the framework of Machine Learning. More specifically, the most widely known and the most commercialized method of application for this task is the Large Language Model. This is basically a term for a machine learning model having many parameters that attempts to model human language. In the present, the underlying model that is used to implement this is the "Transformer". The whole model has been described and explained in its entirety in the paper title "Attention Is All You Need" by Vaswani et. al. It is this model that we are going to understand and implement in this post.

## Foundations

Before we try to approach the model and its parameters and dive into the mathematics of the model, we need to understand what it is exactly that we are doing. We need to know the problem, we need to know why do we aim to solve the problem, we also need to know the solution and we also need to understand the solution.

In the most eagle-eyed sense, we are trying to establish a relation between two sets of words. These two sets of words are very large, differ in size, and sometimes they may even 
