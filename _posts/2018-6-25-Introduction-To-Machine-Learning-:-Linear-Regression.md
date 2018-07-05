---
layout: post
mathjax: true
math: true
published: true
---


![Machine Learning]({{site.baseurl}}/images/LR.png)

In this post we are going to talk about one of the foundational techniques in machine learning, Linear Regression. Linear Regression is used when we want to *fit a line to the data*. We are going to first look at some theory underlying the algorithm, then get some data and code our algorithm!

Let us get the first thing out of the way, i.e., getting the data. We are going to code the algorithm as we go. So, fire up a text editor, and let's get started!

We are going to learn about the basics of linear regression while working on the Boston Housing Dataset. This dataset is small enough to not cause any computational slowdowns but is big enough to help us get started. It comes pre-installed with the sklearn package that we downloaded earlier. So, now we have to import the dataset. We do this as follows:

```python
import sklearn.datasets
boston_data = sklearn.datasets.load_boston()
```

Let us talk about some basics. First of all, Machine Learning, as we talked about, involves learning from data. More precisely, it involves **generalizing** from data. That data is divided into two parts: Train Data and Test Data. As the names suggests, when we train our algorithm, or when the algorithm is *'learning'*, we use the Train Data. 

Then, to test how well the algorithm did on the data, we give it *new, or unseen data*. That is the test data. 

Some last bit of terminology, the data itself, regardless of whether it is Train or Test Data, it is divided into two parts again: *Features* and *Labels*. Consider an example: We are trying to predict the amount of some drug required for a particular disease. 

We have 500 samples, each from a different patient. In this case, each sample is called an *'instance'*. For each instance, let us assume we are using the three following variables to predict the amount:

**1. Age of the Patient**

**2. Height of the Patient**

**3. Weight of the Patient**

In this case, the *Age, Height and Weight* are called the *'Features'*, and the amount of drug that we are trying to predict is called *'Label'*. This is the basic terminology we need for now. Let us extract the *Features and Labels* from the Boston Dataset now. We do it with the following code, along with explanations below it:


```python
features = boston_data['data']
labels = boston_data['target']
train_features = features[:400]
train_labels = labels[:400]
test_features = features[400:]
test_labels = labels[400:]
```


`boston_data` is a *dictionary* that can be indexed with keys and values just as we have seen. The key *'data'* gets the actual data, as a numpy array. The key *'target'* gets the labels. The next four lines perform an operation known as *'slicing'*. This is similar to what the name suggests, it slices an array from the starting point to the ending point. 

So, the code `features[:400]` starts from *instance 0* and goes upto but not including *instance 400*. Similar is the case with *labels*. For test data, we start with *instance 400* and go upto the end. Notice how omitting *(not entering any number)* the start automatically assumes the beginning on the array, whereas omitting the end assumes the end of the array. 


The process we just did above was a small part of what is known as *Data Processing*. Now, we talk about the actual algorithm below.
In machine learning, to *'learn'*, we need to look at how *wrong* we are. To do that mathematically, we define an error function. Let us assume we have *N samples or N instances (eg: N patients, say N = 100)*.

Then if we let $\alpha$ denote our answer, or *'prediction'*, and $ t $ denote the true answer, which is available in the training data at training time, then let us call $y_n - t_n$ be called the **'error'**.
So that $(y_n - t_n)^2$ is the squared error. Where the subscript *n* denotes the *n'th* training sample, say n = 50. If we sum over all the *n* samples in the training data, we get the following formulation:

$$\frac{1}{2}\sum_{i=0}^N (y_n - t_n)^2$$

This is called the Mean Squared Error, or *MSE*. This gives us a sense of ***"How wrong we are"*. And it is this sense, or more formally, this function that we must minimize**.
