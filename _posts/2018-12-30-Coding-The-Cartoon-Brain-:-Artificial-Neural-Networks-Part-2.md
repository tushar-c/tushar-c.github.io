---
published: true
---

![ANN2]({{site.baseurl}}/images/brain.jpg)

In the previous post we talked about Artifical Neural Networks as cartoon models of the brain, in this post we're going to write a neural network from scratch with just numpy based on the expressions derived in the previous section. Let's get straight to it!


## Some Setup

Before we begin, we will need some helper functions, to reduce our workload, and at the same time, to work reliably under numerical instability, as is often the case when doing large-scale mathematical problems with computers.

For our purposes, we need only one script that we will just save into the same directory where we are going to create the file that will contain our code for this post. That required script can be found [here.](https://github.com/tushar-c/MLProjects/blob/master/conv_utils.py)

## Some Generalization

In the previous post, we derived the backpropagation procedure in the non-vectorized form. However, numerical packages such as numpy make it very easy and efficient to implement these operations in vector form. We are going to understand that generalization, and then we will implement the final version in code.

The vectorized form of backpropagation is really again just the chain rule along with clever tricks from matrix calculus and linear algebra. We're going to understand all of that here!

## Vectorizing Backpropagation

If we look at the equations for the forward pass, we can spot some patterns (pun intended!), and get some hints for the backward pass as well. Let's see what they are, by recalling the equations.


$$ a_j = \sum_{i} (w_{ij} * o_{i}) + b_j$$

and

$$ o_j = f(a_j)$$

Seeing the above two equations, we can see that $$a_j$$ is really just a *dot product* between the vectors $$ w_i $$ and $$ o_i $$, to which a vector $$b$$ is added. Remember that vector addition is element-wise.

Also, $$f$$ is again the sigmoid function applied element-wise. Now imagine if each $$ w_i $$ was sitting in the row of a matrix called $$W$$, and all the $$o_i$$ were stacked into a vector called $$o$$, then the above equations would become:

$$ a_k = W_k o_{k - 1} $$

and

$$ o_k = f(a_k) $$

where W_k is the matrix we were talking about, with each $$w_{ij}$$ in the matrix connected weights from the $$i-th$$ neuron in the current layer ($$k$$) to the $$j-th$$ neuron in the previous layer ($$k-1$$).

This is just our forward pass! So that if there are $$L$$ layers in the network, we can do the following for $$l = 1, ..., L$$:

$$ a_l = W_l o_{l - 1} $$

and

$$ o_l = f(a_l) $$

and then for the final output, $$ y = o_l$$, where $$ y $$ is our prediction and the ***MSE(Mean Squared Error)*** in the vectorized form becomes:

$$ L(y_, y) = \frac{1}{2}(y_ - y)^T (y_ - y) $$

where $$ y_ $$ is our prediction, $$y$$ is the true label. Keep in mind that $$L(y_, y)$$ is a scalar, or a real number.




## The Code

```
import numpy as np 
import conv_utils


def make_net(layers):
    return [np.random.randn(w[1], w[0]) for w in layers]


def forward(x, y, weights, f):
    affines = []
    transforms = [x]
    a = x 
    for w in range(len(weights)):
        a = np.matmul(weights[w], a)
        affines.append(a)
        a = f(a)
        transforms.append(a)
    return affines, transforms


def backward(x, y, error, weights, affines, transforms, grad_f):
    grads = []
    g = error
    N = len(weights)
    for n in range(N - 1, -1, -1):
        g = g * grad_f(affines[n])
        grad = np.matmul(g, transforms[n].T)
        grads.append(grad)
        g = np.matmul(weights[n].T, g)
    return [j for j in reversed(grads)]


def sgd(weights, grad_weights, eps):
    for n in range(len(weights)):
        weights[n] = weights[n] - (grad_weights[n] * eps)
    return weights


def mse(y, pred):
    return np.sum(np.power(y - pred, 2)) / 2


N = 50
eps = 0.05
EPOCHS = 100

weights = make_net([(10, 60), (60, 140), (140, 50), (50, 10), (10, 1)])
samples = [np.random.randn(10, 1) for _ in range(N)]
labels = [np.random.randn(1, 1) for _ in range(N)]

f = conv_utils.stable_sigmoid
grad_f = conv_utils.sigmoid_gradient

for e in range(EPOCHS):
    epoch_loss = 0.
    for s in range(N):
        feature, label = samples[s], labels[s]
        a_s, h_s = forward(feature, label, weights, f)

        pred = h_s[-1]
        error = pred - label

        epoch_loss += mse(label, pred)

        weight_grads = backward(feature, label, error, weights, a_s, h_s, grad_f)
        weights = sgd(weights, weight_grads, eps)

    print('Epoch {} / {}; Loss {}'.format(e + 1, EPOCHS, epoch_loss / N))
```

