---
layout: post
mathjax: true
math: true
published: true
---

![Classification]({{site.baseurl}}/images/SheepClass.jpg)

## Introduction To Logistic Regression

In the last post we talked about Linear Regression, which is a technique of predicting numerical (more precisely, continuous) values given an input. Now, even though the name of the technique is Logistic Regression, it involves classification rather than regression. Classification, as the name suggests, involves placing a given input into of **1 of N** classes. 

For eg: Predicting if a mail is spam or not; on 5 varying degrees of severity, which degree of a heart attack chance a patient has, etc.

When $$N = 2$$, we call it *binary classification*. Generally, for $$N > 3$$, we call the scenario *multiclass classification*. In this post we will consider the problem of binary classification. At the end, we will look at just one modification that allows to extend our algorithm to multiple classes.

Recall that last time our predictions were given by $$y = {w^T}x + b$$. This was a continuous value. To extend this same prediction to predict one of two classes, we introduce a function that takes in any value and returns a value in the interval $$[0, 1]$$. This function is heavily used in machine learning and it is the **logistic sigmoid function**.

## Star Of The Show

The logistic sigmoid function is given by the formula:

$$ f(x) = \frac{1}{1 + e^{-x}}$$ 

or equivalently, by:

$$ f(x) = \frac{e^x}{1 + e^x}$$

Let's see what this function looks like, here it is:

![logistic_sigmoid]({{site.baseurl}}/images/sigmoid.png)

See how it has an *S-shape*, it is a very useful property of this function.

So, what does the property of always being in the interval $$[0,1]$$ mean to us? Well, it means that we can assign labels, or *class numbers* with the help of a threshold (usually = $$0.5$$). So that if $$y < 0.5$$, we assign class 0. Else if $$y >= 0.5$$, we assign class 1. But how can we use this function to make predictions by learning from data? We see this below.

## Posing The Problem

Remember that last time the error function we minimized was the sum-of-squares error function given by:

$$\frac{1}{2}\sum_{n=1}^N (y_n - t_n)^2 $$

Where $$t_n$$ is the label for the n-th training example and $$y_n$$ is our prediction for the n-th training example.

In the case of the classification problem, it is easy to see that this function is not *descriptive* as we would like. As now the labels and our predictions are going to be one of two possible values, either $$0$$ or $$1$$. So that the terms in the above sum will be either $$0$$ or $$1$$ (make sure you convince yourself of this). We therefore use a different function to measure the error.

This function is called the *cross-entropy* error function and is given by:

$$-\sum_{n=1}^N {t_n}\log(y_n) + {1 - t_n}\log(1 - y_n)$$

Here the notation is the same as above, except that $$y_n = f($${w^T}x + b$$)$$ and $$f$$ is the logistic sigmoid function. 

The following small optional section describes how to obtain this function as a measure of error. Even if you do decide to skip it, the key thing to take away is the following insight: 

**In Machine Learning, it is always a good idea to write the error as the negative of our probability assignment function. This is because minimizing the negative of something corresponds to maximizing its positive. So that when our error function, so defined, is minimized, our probability function tends to assign high probabilities to correct classes.**

## (OPTIONAL) Derivation Of The Cross-Entropy Function:

We start by noting that each of our prediction is going to be in one of two classes. So that we can write our probability as:

$$p(x_n) = {y_n}^{t_n}{1 - y_n}^{1 - t_n}$$

where $$t_n$$ is either $$0$$ or $$1$$, $$y_n$$ is our prediction. We write down the *Likelihood function*, this is the product of the above term, just for all $$N$$ terms in the dataset:

$$ L(X) = \prod_{n=1}^N p(x_n) = \product_{n=1}^N {y_n}^(t_n){1 - y_n}^(1 - t_n) $$



The problem to pose here is going to an optimization problem, similar to the one in the last post. But we arrive at it through a slighly different route. Namely, through the idea of Maximum Likelihood. This method looks to maximize the possibility of the occurence of the optimum value of some variable. We call it *optimizing the objective function with respect to this variable*. Here is the method:

