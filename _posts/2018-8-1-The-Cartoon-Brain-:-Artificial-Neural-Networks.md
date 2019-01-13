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

$$ y = \sum_{n=1}^3 (x_n * w_n) $$

where the $$ w_n $$ , $$n = 1, 2, 3$$ denote the weights and similary the $$x_n$$ denote the inputs. This simple computation is really the basis for most ANNs.

But notice how this computation is *linear*, in fact, the **activation function** $$f$$ as it is so called, is the identity in this case, which means:

$$ f(x) = x$$

Using this we can write the computation in the above image as:

$$ y = f(\sum_{n=1}^3 (x_n * w_n)) $$



