---
published: true
---

![ANN2]({{site.baseurl}}/images/brain.jpg)

In the previous post we talked about Artifical Neural Networks as cartoon models of the brain, in this post we're going to write a neural network from scratch with just numpy based on the expressions derived in the previous section. Let's get straight to it!


## Some Setup

Before we begin, we will need some helper functions, to reduce our workload, and at the same time, to work reliably under numerical instability, as is often the case when doing large-scale mathematical problems with computers.

For our purposes, we need only one script that we will just save into the same directory where we are going to create the file that will contain our code for this post. That required script can be found [here].(https://github.com/tushar-c/MLProjects/blob/master/conv_utils.py)

## Some Generalization

In the previous post, we derived the backpropagation procedure in the non-vectorized form. However, numerical packages such as numpy make it very easy and efficient to implement these operations in vector form. We are going to understand that generalization, and then we will implement the final version in code.

The vectorized form of backpropagation is really again just the chain rule along with clever tricks from matrix calculus and linear algebra. We're going to understand all of that here!

## The Code

```
import numpy as np 
import conv_utils


def make_net(layers):
    return [np.random.randn(w[0], w[1]) for w in layers]


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

weights = make_net([(100, 10), (150, 100), (80, 150), (40, 80), (10, 40), (2, 10)])
samples = [np.random.randn(10, 1) for _ in range(N)]
labels = [np.random.randn(2, 1) for _ in range(N)]

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

