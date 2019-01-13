---
published: true
---
![Classification]({{site.baseurl}}/images/Bio_Ann.png)

## Introduction
In this post we are going to talk about a learning technique that is, at least at the very minimum, based on the structure of a brain. The title of the blog post summarizes this approach. 

While at first sight it might seem that with the terminology of *neurons* and *neuronal weights* we are modelling something like the brain, in reality, ANNs (Artificial Neural Networks) are really just complex non-linear (possibly linear) pattern recognition architectures. 

## A Few Points To Keep In Mind
Before we begin describing the architecture and how it works, it is important to keep in mind *what the algorithms are really doing*. It is very easy to get hung up into the machinery involved and lose sight of the *big picture*. 

As noted at the start, these models are really *cartoon* models (oversimplifications) of the actual neurons in our brains. While these models work very well in many domains in practice, their applicability in understanding the neurophysiological basis of things is limited.

In this post we will use the term **neuron** to refer to an **artifical neuron**.


## Let's Begin
![Classification]({{site.baseurl}}/images/1_tMuOsWWRX3fR84xoSeJcAw.png)

As the above image shows, in the most simple case, an artificial neuron has the basic computational structure of computing weighted sums of its inputs. In the above image, the output **y** is obtained from its inputs is obtained as follows:

$$ y = \sum_{n=1}^3 (x_n * w_n) + b $$

where the $$ w_n $$ , $$n = 1, 2, 3$$ denote the weights and similary the $$x_n$$ denote the inputs, the $$b$$ parameter denotes the *bias*, or the *offset*. This simple computation is really the basis for most ANNs.

But notice how this computation is *linear*, in fact, the **activation function** $$f$$ as it is so called, is the identity in this case, which means:

$$ f(x) = x$$

Using this we can write the computation in the above image as:

$$ y = f(\sum_{n=1}^3 (x_n * w_n)) = \sum_{n=1}^3 (x_n * w_n) + b$$

But *linearity* is boring, in fact, reality is *non-linear* and *noisy*. Here, we introduce *non-linearities* into the system by using other forms of $$f$$ as opposed to the identity.

One *non-linearity* we have already seen is the *sigmoid* function in the post about logistic regression. There are other such functions used too, such as *tanh*, *relu*, etc. In fact, you can use a function of your own as long it is well behaved, code for *differentiable*.

The property of *differentiability* is going to come in handy later.

The logistic sigmoid is given by:

$$ f(x) = \frac{1}{1 + e^{-x}}$$ 

In our case, we can then write the computation of this neuron as:

$$ y = f(\sum_{n=1}^3 (x_n * w_n)) = \frac{1}{1 + e^{\sum_{n=1}^3 (x_n * w_n) + b}}$$

## Multiple Neurons

Now that we've seen the computation for a single *neuron*, you might think that the computational power of such a system is limited and indeed this is the case. However, we can easily this to multiple neurons, such as shown in the below image:

![Classification]({{site.baseurl}}/images/tikz10.png)


The first *layer* that you see is just the inputs, while the actual neurons are in the *first hidden layer*, where each neuron performs the computation defined by $$y$$ just above, and the final *layer* is the *output layer*.

In this image, the *input layer* has three inputs, let us call them $$x_i, i = 1, 2, 3$$. The *hidden layer* has four neurons, let us call them $$n_i, i = 1, 2, 3, 4$$. The output is just one neuron, let us call it $$o$$.

Thus we can describe the computation of each $$n_i$$ as shown below:

$$ y = f(\sum_{n=1}^3 (x_n * w_n)) = \frac{1}{1 + e^{\sum_{n=1}^3 (x_n * w_n)}}$$

