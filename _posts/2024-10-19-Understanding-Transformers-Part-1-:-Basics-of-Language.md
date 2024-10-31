![ANN2]({{site.baseurl}}/images/humanLanguage.jpg)

## Introduction

After some time away, we are back! This time, we are going to understand the idea of modelling human language through the framework of Machine Learning. More specifically, in this first of a five-part series of blog posts, we are going to try to understand and implement the most widely known and the most commercialized method of application for this task is the Large Language Model. 

This is basically a term for a machine learning model having many parameters that attempts to model human (or natural) language.

Presently, the underlying model that is used to implement this is the **'Transformer'**. The whole model has been described and explained in its entirety in the paper title **'Attention Is All You Need' by Vaswani et. al**. You can find the paper [here.](https://arxiv.org/abs/1706.03762)

It is this model that we are going to understand and implement in this post. If you want to skip to the whole code, you can find it [here.](https://gist.github.com/tushar-c/d0c6b51822f1daf067b51cb68788acd7)

## Foundations

Before we try to approach the Transformer Model and its parameters and dive into the mathematics of the model, we need to understand what it is exactly that we are doing.

In the most eagle-eyed sense, **we are trying to get the computer to establish a relation between two gigantic sets of words**. These two sets of words are very large, differ in size, and sometimes they may even belong to different languages. We aim to give the computer a set of words as input and we are then asking different questions such as:

1. Given this set of words, what is the most probable set of words that come next in the sequence?

2. Given this set of words (in one language), what is the most probable set of words in another given 		  language?

These are just two example questions that we may ask the computer. We can ask even more questions! However, for now, this will have serve as an exhaustive list of the curiousity of the human mind when it comes to asking natural language questions to computers. As you may have understood, given the current sequence of words, the computer is going to output **probabilities** of the next words in the sequence.

Now, given that computers only understand binary and the best method we have of communicating directly with them is through programming languages, how do we feed everyday language to a computer and expect it to give us everyday language as output that makes sense?

The answer is a little 'trick' that we have developed called **'tokenization'**. Understanding this technique is very simple if we think of it in the following manner (we will take the English Language for our purposes):

1. We take any given word in english. We then decide a large number. This number is meant to be a total of all the words that our model will ever see. We call this term the **'vocabulary'**. Similarly to our interpretation of the word in everyday life, the **vocabulary** is the total words that our model will ever come across. 

2. Now, for each word in our input, we create a **vocabulary-dimensional, one-hot vector**. This means that if our vocabulary has _1000 words_, then we will create a _1000-dimensional vector_ for each word in our input that we feed to a computer. We need to do one more thing. Since there is the _'one-hot vector'_ term, we need to make sure that each vector is a _one-hot vector_. 

How do we do this? Well, it's simple. For each word in our vocabulary, we create a vector and every position in the vector is set to zero except for the position that corresponds to the position of the current word in our vocabulary.

For example, if our input is _"The sun shines"_, then we will have 3 vectors, each vector corresponding to a word in the input, and each vector will be a _1000-dimensional, one-hot vector_, since our _"English Vocabulary"_ for the purposes of this tutorial has only _1000 words_.

3. To illustrate this with an example, suppose we continue with our above example of _"The sun shines"_, we would need to create 3 vectors, one for each word, and then convert each vector to a _one-hot vector_ based on the position of that word in our overall vocabulary. For the sentence _"The sun shines"_, we will have the following:

$$  The  \begin{bmatrix} 1 \cr 0 \cr 0 \end{bmatrix} $$

$$  Sun  \begin{bmatrix} 0 \cr 1 \cr 0 \end{bmatrix} $$ 

$$  Shines  \begin{bmatrix} 0 \cr 0 \cr 1 \end{bmatrix} $$


We assume that _'The'_ is the first word in vocabulary, _'sun'_ is the second word in vocabulary and _'shines'_ is the third in our vocabulary. We have assumed that there are 3 words in our vocabulary. If there were 5, we would have three 5-dimensional vectors for a sentence of 3 words in a vocabulary of a total of 5 words.

If you made it this far, congratulations! We need to understand just one more concept before we can begin with the implementation of the transformer and that is, the **'Embedding Matrix'**.

Understanding the _Embedding Matrix_ is really simple. The Embedding Matrix basically takes our one-hot vectors and projects them into real-valued vectors so that these may be fed into our Neural Networks and then the Network may adapt its weights through the backpropagation algorithm in the learning phase. 

The basic idea of the Embedding Matrix is to give _real-value continuous representations_ to human language words (in this case, English). By combining the concepts of one-hot vectors and the Embedding Matrix, we are giving a real-valued continuous representations to words pertaining to human languages. 

These numerical representations of human language words let us train the Neural Networks on our inputs.

Let's say that our first layer of the Neural Network takes a 100-Dimensional Vector as input and outputs a 256-Dimensional Vector so that it may be fed into the next layer of the network, then that means that the Weight Matrix for that layer would a $$ 100 \times 256 $$-Dimensional Matrix.

Similarly, let's take the example of our 3-Dimensional Vectors from our vocabulary above. Imagine if the layer takes a 6-Dimensional Vector as input and outputs a 12-Dimensional or any other N-Dimensional Vector to further project into the Neural Network. Therefore, the Embedding Matrix that we are looking at is a $$ 3 \times 6 $$-Dimensional Matrix.

These are the basics we need to begin our implementation of the Transformer Model.

## Conclusion

Now that we know these basics, we can move ahead towards understanding and implementing the Transformer Model in the next post!
