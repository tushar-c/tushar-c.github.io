---
layout: post
mathjax: true
math: true
published: true
---

![Classification]({{site.baseurl}}/images/SheepClass.jpg)

In the last post we talked about Linear Regression, which is a technique of predicting numerical (more precisely, continuous) values given an input. Now, even though the name of the technique is Logistic Regression, it involves classification rather than regression. Classification, as the name suggests, involves placing a given input into of **1 of N** classes. 

For eg: Predicting if a mail is spam or not; on 5 varying degrees of severity, which degree of a heart attack chance a patient has, etc.

When $$N = 2$$, we call it *binary classification*. Generally, for $$N > 3$$, we call the scenario *multiclass classification*. In this post we will consider the problem of binary classification. At the end, we will look at just one modification that allows to extend our algorithm to multiple classes.

Recall that last time our predictions were given by $$y = {w^T}x + b$$. This was a continuous value. To extend this same prediction to predict one of two classes, we introduce a function that takes in any value and returns a value in the interval $$[0, 1]$$. This function is heavily used in machine learning and it is the **logistic sigmoid function**.

This function is given by the formula:

$$ f(x) = \frac{1}{1 + e^{-x}}$$ 

or equivalently, by:

$$ f(x) = \frac{e^x}{1 + e^x}$$

Let's see what this function looks like, here it is:

![logistic_sigmoid]({{site.baseurl}}/images/sigmoid.png)

See how it has an *S-shape*, it is a very useful property of this function.


