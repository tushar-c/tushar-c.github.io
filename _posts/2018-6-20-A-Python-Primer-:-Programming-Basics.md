---
published: false
---
![SoftwareDev]({{site.baseurl}}/images/Python-programming-for-hackers-compressed.jpg)

Hello! In this post we are going to go over the parts of the Python Programming Language and of the software packages (which we installed last time) that will be most relevant to us. If you are interested in learning more of the language and the libraries, links will be provided at the end.

Let's start! Fire up your text editor (or open cmd and type 'python'), and follow along. We will cover variables, loops, functions, some basic data structures like lists, dictionaries and sets. We will also use basic functions of the numpy library and finally close with some basic plotting with matplotlib. All this will be extremely useful starting from the next post when we start working with actual biological data.

If you are working in cmd, then typing in *python* should give something similar to this:

```python
tushar@home:~$ python
Python 3.6.5 (default, Apr  1 2018, 05:46:30) 
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```
Ofcourse the exact output might differ, what's important is you should see the prompt (**'>>>'**) at the end.

##VARIABLES##

A variable, as the name suggests, are objects that are dynamic rather than static. This means the value they store can be changed. A simple example is:

```python
>>> x = 25
```
This simple example highlights one of the most important features of Python -- simplicity and readability. This command stores the integer value 25 in x. To see this, type in:

```python
>>> x
25
>>> 
```

Simple addition, subtraction, multiplication, division and exponentiation (raising a number to some power) can be done as shown:

```python
>>> y = x + 13
>>> x
25
>>> y
38
>>> y - x
13
>>> y + x
63
>>> y * x
950
>>> y / x
1.52
>>> y ** x # exponentiation, raising y to the power of x
3123128945369882154942078678703504621568
>>> 
```

Comments in Python start with '*#*', everything after this Python does not interpret as code and ignores it. This completes our requirements of Python variables. We next take a look at loops.

##LOOPS##

As the name name suggests, a loop is way of doing some operation repetitively until some condition is true, or for a fixed number of times, or even forever! There are 2 major types of loops, the **'for loop'** and the **'while loop'**.

**The 'for' Loop**

We will first look at a for loop. A for loop does an operation some fixed number of times as specified by us and then once it has completed that step said number of times, it stops. In Python, a for loop behaves a little differently than some other languages and it can be used in a variety of ways. We will see 2 most common ways. An example is given below:

```python

# a simple for loop

>>> counter = 10
>>> for c in range(counter):
...     print(c)
... 
0
1
2
3
4
5
6
7
8
9
>>> 
```
The variable *'c'* starts from the value 0 and goes upto (but not including) 9.

One thing to point out, **range** is a function that does something similar to the literal meaning of its name. **range** starts from a beginning variable (usually 0, but we can set it to anything we like, see example below) and goes upto (but not including) the value we give it (in our case 10). We will talk more about functions below.

```python

# another simple for loop

>>> counter = 10
>>> for c in range(2, counter):
...     print(c)
... 
2
3
4
5
6
7
8
9
>>> 
```

A for loop can also act in a *reversed* manner. That means, in our case, we could start from 10 and go down to 0. Here's how:

```python
>>> for c in range(counter, -1, -1):
...     print(c)
... 
10
9
8
7
6
5
4
3
2
1
0
>>> 
```
Here, the first value is 10 (equal to our variable *'counter'*), the second value is stop value (equal to -1, because the loop goes upto but does not include the -1. As an exercise, see what happens if you put 0 instead of -1). The third value is the number by which we should decrease our current value of the variable *'c'*.

**The 'while' loop**
