---
layout: post
mathjax: true
math: true
published: true
---


![Machine Learning]({{site.baseurl}}/images/LR.png)

In this post we are going to talk about one of the foundational techniques in machine learning, Linear Regression. Linear Regression is used when we want to *fit a line to the data*. The term regression refers to the fact that we are predicting *continuous values* like prices, amounts, etc. rather than *discrete values* such as categories, types, etc. We are going to look at some theory underlying the algorithm, get some data and code our algorithm!

Let us get the first thing out of the way, i.e., getting the data. We are going to code the algorithm as we go. So, fire up a text editor, and let's get started!

We are going to learn about the basics of linear regression while working on the Boston Housing Dataset. This dataset is small enough to not cause any computational slowdowns but is big enough to help us get started. It comes pre-installed with the sklearn package that we downloaded earlier. So, now we have to import the dataset. We do this as follows:

```python
import sklearn.datasets
boston_data = sklearn.datasets.load_boston()
```

Let us talk about some basics. First of all, Machine Learning, as we talked about, involves learning from data. More precisely, it involves **generalizing** from data. This means predicting values of **previously unseen data, or data that we did not use when training the algorithm**. That data is divided into two parts: Train Data and Test Data. 

As the names suggests, when we train our algorithm, or when the algorithm is *'learning'*, we use the Train Data. Then, to test how well the algorithm did on the data, we give it *new, or unseen data*. That is the test data. 

Some last bit of terminology, the data itself, regardless of whether it is Train or Test Data, is divided into two parts again: *Features* and *Labels*. Consider an example: We are trying to predict the amount of some drug required for a particular disease. 

We have 500 samples, each from a different patient. In this case, each sample is called an *'instance'*. For each instance, let us assume we are using the three following variables to predict the amount:

**1. Age of the Patient**

**2. Height of the Patient**

**3. Weight of the Patient**

In this case, the *Age, Height and Weight* are called the *'Features'*, and the amount of drug that we are trying to predict is called *'Label'*. This is the basic terminology we need for now.

## Linear Regression on the Boston Housing Data


![Boston Housing]({{site.baseurl}}/images/houses.jpg)


Let us extract the *Features and Labels* from the Boston Dataset now. We do it with the following code, along with explanations below it:


```python
features = boston_data['data']
labels = boston_data['target']
train_features = features[:400]
train_labels = labels[:400]
test_features = features[400:]
test_labels = labels[400:]
```


`boston_data` is a *dictionary* that can be indexed with keys and values just as we have seen in previous tutorials. The key *'data'* gets the actual data, as a numpy array. The key *'target'* gets the labels. 

The next four lines perform an operation known as *'slicing'*. This is similar to what the name suggests, it slices an array from the starting point to the ending point. 

So, the code `features[:400]` starts from *instance 0* and goes upto but not including *instance 400*. Similar is the case with *labels*. For test data, we start with *instance 400* and go upto the end. Notice how omitting *(not entering any number)* the start automatically assumes the beginning on the array, whereas omitting the end assumes the end of the array. 


The process we just did above was a small part of what is known as *Data Processing*. Now, we talk about the actual algorithm below.
In machine learning, to *'learn'*, we need to look at how *wrong* we are. To do that mathematically, we define an error function. Let us assume we have *N samples or N instances (eg: N patients, say N = 100)*.

If we let $$y_n$$ denote our answer, or *'prediction'*, and $$t_n$$ denote the true answer, which is available in the training data at training time, then let us call $$y_n - t_n$$ be called the **'error'**.
So that $$(y_n - t_n)^2$$ is the squared error. Where the subscript *n* denotes the *n'th* training sample, say n = 50. If we sum over all the *N* samples, starting from *1* and going upto *N* in the training data, we get the following formulation:

$$\frac{1}{2}\sum_{i=1}^N (y_n - t_n)^2$$

This is called the Mean Squared Error, or *MSE*. This is one of many functions that gives us a sense of ***"How wrong we are"*. And it is this sense, or more formally, this function that we must minimize**.


## Posing The Problem

As we saw above, the problem is to minimize how wrong we are. To do this on a computer, we pose a function, a mathematical mapping that takes an input and returns an output, and we aim to minimize it.

That function is:

$$\frac{1}{2}\sum_{i=1}^N (y_n - t_n)^2$$

The meaning of the individual symbols is given above. We said $$y_n$$ is our prediction, but how do we generate it? This the core question answered differently by different machine learning algorithms. For our purposes in this post, we are going to do that by taking what is called the *dot product* between two vectors. We explain each of these below:

One vector would contain our features for one instance (or one sample in the dataset). *For eg: in our patient example above, each vector x would be a 3 x 1 vector, i.e., 3 rows and 1 column, with each row representing one of the following: Age , Height and Weight. So the vector would look something like:

$$ \left[ 
\begin{array}
	AAge\\
  	Height\\
    Weight
\end{array}
\right]
$$

So if *Age = 20, Height = 179 cm, and Weight = 75 kgs*, then the vector will look like:

$$ \left[ 
\begin{array}{cc}
	20\\
  	179\\
    75
\end{array}
\right]
$$


Since the boston housing data contains 13 *features* for each instances, each vector *x* will be a *13 x 1* vector. To see exactly what those 13 features are, see point 7. and then  following 13 points in this [link.](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names) The 14th point we have to predict .

The other vector is what is called the **parameter vector or the weight vector**. This consists of our 'learned' weights, that when we combine with the instance vector, we get a real-numbered value. Here is a simple picture explaining a dot product:


![Machine Learning]({{site.baseurl}}/images/phpsyWwce.png)


So, we see that a product simply multiplies the corresponding elements in two vectors, does this for all the elements in the two vectors, and finally adds them. Note how two vectors are inputs, and the output is a single number. Here is a mathematical way to write the dot product with our two vectors, *x* and *w*, which represent the instance vector and the weight vector, respoectively:

$$ \sum_{i=1}^N (x_n * w_n) = {w^T}x = {x^T}w $$







