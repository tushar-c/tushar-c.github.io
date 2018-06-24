---
published: true
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
Ofcourse the exact output might differ, what's important is you should see the prompt (**the three '>'**) at the end.

## VARIABLES

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

## LOOPS

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

The *'while'* loop is the other major type of loop that we use in programming. Similar to the *'for'* loop, it runs until a condition is *True rather than upto some fixed value*. An example is shown below (read comments in the code!):


```python
>>> age_till_adult = 18
>>> current_age = 5
>>> while current_age < age_till_adult: # the condition that checks whether to stop or keep going
...     print('I am {} years old.'.format(current_age)) # see explanation below
...     current_age = current_age + 1 # Important to increase the variable, otherwise the loop will run forever!
... 
I am 5 years old.
I am 6 years old.
I am 7 years old.
I am 8 years old.
I am 9 years old.
I am 10 years old.
I am 11 years old.
I am 12 years old.
I am 13 years old.
I am 14 years old.
I am 15 years old.
I am 16 years old.
I am 17 years old.
>>> 
```

The print function (we talk about functions below) above uses a type of '*substitution*', in that the value takes the place of the '*{}*'. This substitution is done by the *.format()* part, which helps us in writing reusable print functions. If you have more than one the curly braces (as they are called), the order in which you pass the variables is the order in which they will appear in the string. An example is shown below:

```python
>>> country_1 = 'Brazil'
>>> country_2 = 'Argentina'
>>> print("{} might win the 2018 FIFA World Cup. However, it's not looking good for {}".format(country_1, country_2))
Brazil might win the 2018 FIFA World Cup. However, it's not looking good for Argentina
>>> 
```

Another thing to point out is the variable update. Instead of: ```current_age = current_age + 1```, we can also use: ```current_age += 1``` , this is easier to read, but a might take some getting used to. The same holds for subtraction, multiplication or division.

That does it for loops, next we talk about functions.


## FUNCTIONS

A function is a utility in programming that allows us to reuse the same block of code. This might happen if you were to perform the same computation, but the input values kept of changing. For example: *Logging some patient data, with each patient having different details or Performing some calculation that involves the same formula, but the inputs keep changing, etc.*  .

A function takes inputs (also called *arguments*) and returns outputs (also called *return values*).

Saying the same thing, but a little more formally, we can write: **Function: Inputs -> Outputs**. In Python, we need 3 components for creating a function:

**1.** The keyword *def* . This tells Python we intend to create a function.

**2.** The name of the function, followed by circular brackets ( *'()'* ), which contain the arguments.

**3.** The return value of the function.

After you have defined a function, you need to *call* it. This is done by typing the function name, then passing the arguments it takes within the circular braces.

Let's put this all together and see an example of a function that prints a nice introduction for you given your first name and then call it.

```python
>>> def introduction_generator(first_name): # define the function
...     introduction = "Hi, my name is {} and I like football!".format(first_name) # do stuff with it
...     return introduction # return value
...
>>> name = 'Tushar'
>>> introduction_generator(name) # call the function
'Hi, my name is Tushar and I like football!' # Isn't that nice!
>>> 
```

Try changing the string *introduction* inside the function. Try changing the argument *first_name* and try passing different values.

As a last point, we can easily extend our function to have more than a single argument. An example follows below:

```python
>>> def introduction_generator(first_name, last_name): # define the function
...     introduction = "Hi, my name is {} {} and I like football!".format(first_name, last_name)
...     return introduction # return value
...
>>> first_name = 'Four-Eyed'
>>> last_name = 'Willy'
>>> introduction_generator(first_name, last_name) # call the function
'Hi, my name is Four-Eyed Willy and I like football!' # :)
>>> 
```

This completes our discussion of functions. One last topic on the Python Basics remains, Data Structures.

## DATA STRUCTURES

**LISTS**

A list is a generalization of a variable. Where a variable stores a single value, a list can store multiple values. Here is an example:

```python
>>> ages = [18, 20, 40, 50, 32, 15]
>>> ages
[18, 20, 40, 50, 32, 15]
>>> 
```

We can lists of other data types such as strings too.

```python
>>> names = ['fred', 'shaggy', 'scooby', 'jerry']
>>> names
['fred', 'shaggy', 'scooby', 'jerry']
>>> 
```
Moreover, we can even mix these data types! Here is another example:

```python
>>> names_and_ages = ['fred', 'shaggy', 18, 'jerry', 20, 10]
>>> names_and_ages
['fred', 'shaggy', 18, 'jerry', 20, 10]
>>> 
```
We can thus mix different data types in a list, in any order.

Let us *iterate* (look at each element one by one) over a list and print its contents.

```python
>>> for entry in names_and_ages:
...     print(entry)
... 
fred
shaggy
18
jerry
20
10
>>> 
```
We can do much more! For links, see the end of the post.

**DICTIONARIES**

As the name would strike a meaning, that very meaning is what a dictionary in Python does. It contains *keys*, each of which corresponds to a *value*, much like in a real dictionary where the *keys* are the words and the *values* their meanings.

Here is an example:

```python
>>> age_dictionary = {'bob':18, 'will':20, 'may':22, 'tuna':2}
>>> age_dictionary
{'bob': 18, 'will': 20, 'may': 22, 'tuna': 2}
>>> 
```
This is a string where names have ages as values. The *:* (colon) indicates a separation. The entries to the left are keys, to the right are values. Each key-value pair is separated by commas.

To access a value (analogous to finding a meaning), we do as shown:

```python
>>> age_dictionary['bob']
18
>>> 
```
also

```python
>>> age_dictionary['tuna']
2
>>> 
```
We can even change the value associated with a key. Here's how:

```python
>>> age_dictionary['tuna'] = 200
>>> age_dictionary['tuna']
200
```
That tuna is really old !

We now talk about sets.

**Sets**

A set is a data structure that does not contain any duplicates. Here is an example:

```python
>>> x = [1,1,2,2,3,3,4,4,5,6] # we first declare a list
>>> x
[1, 1, 2, 2, 3, 3, 4, 4, 5, 6] # It contains duplicates
>>> y = set(x) # we call the set function, passing x as the argument. It returns the set y.
>>> y
{1, 2, 3, 4, 5, 6} # Voila! y does not contain any duplicates.
```

For our purposes, that's all there is to sets really! That completes our discussion of Python basics. We will now talk about some basics we need for numpy and matplotlib. Consider taking a break! :)

## NUMPY BASICS
