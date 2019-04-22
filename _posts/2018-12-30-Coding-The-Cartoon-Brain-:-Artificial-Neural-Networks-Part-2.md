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

Also, $$f$$ is again the sigmoid function applied element-wise. Now imagine if each $$ w_i $$ was sitting in the row of a matrix called $$W$$, and all the $$o_i$$ were stacked into a vector called $$o$$, then, ignoring the bias vector $$ b $$ for now, the above equations would become:

$$ a_k = W_k o_{k - 1} $$

and

$$ o_k = f(a_k) $$

where W_k is the matrix we were talking about, with each $$w_{ij}$$ in the matrix connected weights from the $$i-th$$ neuron in the current layer ($$k$$) to the $$j-th$$ neuron in the previous layer ($$k-1$$).

This is just our forward pass! So that if there are $$L$$ layers in the network, we can do the following for $$l = 1, ..., L$$:

$$ a_l = W_l o_{l - 1} $$

and

$$ o_l = f(a_l) $$

and then for the final output, $$ y = o_l$$, where $$ y $$ is our prediction and the ***MSE(Mean Squared Error)*** in the vectorized form becomes:

$$ L(y, t) = \frac{1}{2}(y - t)^T (y - t) $$

where $$ y $$ is our prediction, $$t$$ is the true label. Keep in mind that $$L(y, t)$$ is a scalar, or a real number.

With these definitions, we can already implement half the code, but first we import two required libraries.

```python
import numpy as np 
import conv_utils
```

`numpy` is there as always, while `conv_utils` refers to the helper script we described earlier.


The forward pass can be defined now as shown.

```python
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
```

Because each layer has a single weight matrix, `len(weights)` gives us the number of layers. `affines` refers to the $$ a_l $$, while transforms refers to $$ o_l $$, we cache both of them. $$f$$ refers to the *activation function* used.

The loss in vectorized form can be written as here.

```
def mse(y, t):
    error = y - t
    loss = np.matmul(error.T, error) / 2
    return loss[0][0]
```

Simple, in line with the equation for the **MSE** above, with the two zero indices present for better presentation in the output.

Through the same tricks used for forward pass, we can derive the equations for the backward pass as follows.

$$ g = \nabla_{y}{L} = (y - t)$$

then for layers $$ l = L, ..., 1$$ we do:

$$ g = g \odot f'(a_l) $$

where $$ \odot $$ is the element-wise multiplication operator for vectors, also known as the hadamard product. 

$$ \nabla_{W_l}{L} = g o_{l - 1}^T $$

this just above are the gradients, now what do we do with the gradients ? We'll see a simple algorithm to use these gradients in one line shortly.

$$ g = W_{l}^T g $$

And these are the backward equations! Now to update the gradients we use a simple and widely used algorithm called Stochastic Gradient Descent (SGD), which is based on a simple equation.

$$ w = w - \epsilon * \nabla{w} $$

where $$ \epsilon $$, *epsilon*, is also referred to as the *learning rate*, usually a constant, but it can be a variable quantity.

SGD can be implemented as shown below.

```
def sgd(weights, grad_weights, eps):
    for n in range(len(weights)):
        weights[n] = weights[n] - (grad_weights[n] * eps)
    return weights
```

With that, the final function is the backpropagation method as shown below.

```
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

```

Look at the code carefully, the code makes `grad` use `transforms[n]` while in the equation we had

$$ \nabla_{W_l}{L} = g o_{l - 1}^T $$

This is because `transforms` has one more element than `affines`, and that is the original input `x`.

We also create random *10-dimensional data vectors* as inputs, and *real value random scalars* as labels.

`EPOCHS` refers to the number of times we go through the whole dataset.


We show the full code below.


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


def mse(y, t):
    error = y - t
    loss = np.sum(error.T, error) / 2
    return loss[0][0]


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


In this, `make_net` initializes the `weights`. Take not how `make_net` swaps the elements of the tuple passed to it as the arguments, this is because when we call `make_net` in the definition of `weights`, we use tuples of the form (`input_dimension`, `output_dimension`).

## Conclusion

And that's it! That is the backpropagation algorithm in code. Go through it more if you feel that you need to study this more, as it can take time to get used to this. 

Next time we'll see a very exciting and applied field (it's a surprise) that has great potential and where Machine Learning and Signal Processing have great applications.

I'll see you next time!
