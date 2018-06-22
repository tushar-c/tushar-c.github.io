---
published: true
---
In this post we are going to set up our software environment so that we can can run the algorithms on our computers. The reason is that while understanding is crucial, you should also *see the algorithm in action*. 

Much like spending all your time reading everything you can about cars but never really driving one seems like a loss, so is reading everything about the algorithm but never seeing it written out in code. To this end, we require 3 major things apart from a computer with an internet connection, viz. :

1. A programming language to express our ideas

2. A text editor to write our code in

3. Software Packages which will help us in not reinventing the wheel (think about how annoying it would be if you had to write your own programs defining addition, subtraction, multiplication, matrix multiplications, and so on.) This step is not hard at all as you will see now.

Okay, so the steps to setup are really easy and I am going to assume the user has the Windows Operating System installed (Linux steps will be described at the very end, the steps for Mac OS are very similar).


Step 1. Head to the [Official Python Website](https://www.python.org/downloads/windows/ "Official Python Website"), click the very first entry that says "Latest Python 3 Release - Python 3.6.5". Now, scroll to the end, and there choose the **Windows x86-64 executable installer** if you have a 64-bit system or else choose the **Windows x86 executable installer** if you are on a 32-bit system. 

If you do not know how to check your version, here is a link that helps -> [Determining Windows Version](https://support.microsoft.com/en-in/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) . If you still cannot determine, Youtube has a plethora of these. Then once you download the installer, install it like any other windows program by following the on-screen instructions. 

*Note:* While installing, if there occur any errors, Google is your best friend. Usually on Windows, most Python installation errors seem to be fixed by going to the 'Windows Updates' tab and just updating the packages there.


Step 2. If you already use a text editor (such as Atom, Sublime Text, etc.), then skip this step. Another popular choice is Notepad++, this is much more rich than the default Notepad Application that comes with Windows. Download and install it from here, again, according to whether you have a 32-bit or a 64-bit system. Link -> [Notepad++ Download Link](https://notepad-plus-plus.org/download/v7.5.6.html). Install it like any other Windows program.


Step 3. If you have managed to do the above steps, only 1 more step remains on your pc, open up the command prompt (abbreviated as 'cmd'). This step involves typing commands but all of them are the same with just the name of the package different. We will install a total of 4 packages. Follow along, and everything should work out fine!

Step 3 (Continued). Once you have the cmd open, type in the following commands:

```python
pip install numpy
```
Press Enter now and wait for it to complete. Pip is a program installer, so when we say: *pip install (package name)*, we ask pip to go download some software for us and then install it. The rest of the packages are similar to install!

Type in the following:

```python
pip install matplotlib
```

Wait for this to finish. 2 more to go!

```python
pip install tensorflow
```

```python
pip install keras
```

I should point out that the point of downloading these packages is to ease our work as well as to be able to run the programs made beforehand for demonstration and experimentation.

Also, the packages you downloaded were specifically designed for these purposes: numpy is for numerical computation such as matrix multiplications, etc. , matplotlib is for plotting graphs and visualizing, tensorflow is a mathematics library made especially for machine learning and deep learning, keras makes tensorflow easy.

**NOTE: In technical terms, when a program requires some other packages to be installed to function (as it may use some of their features), those packages are called dependencies.**


And you're all set! You have the programming language, the text editor, and the required software packages. Congratulations!


*PARTING NOTE: While my goal is to make things easy as this demo might have pointed out, the reader will also have to follow along and be prepared to fix things, search for errors on Google on their own.StackOverflow is also a great place to look for answers to your questions. * 

*This is so that you can have a small introduction to the whole process of IDEA -> IMPLEMENTATION -> DEBUGGING -> DEPLOYMENT, which is the Computer Programming equivalent of The Central Dogma in Molecular Biology.*

**DISCLAIMER: Read Only If Installing On Linux (Debian-based systems, eg:Ubuntu)**

**Step 1. Python usually comes pre-installed as python3, otherwise you can download from their official website.**

**Step 2. Gedit is a nice text editor. You can always use Sublime Text or Atom from their official websites.**

**Step 3. Type in the same commands as mentioned above, though you might have to provide superuser privileges, eg: sudo pip install (package name); you might also have to install pip for python3 itself, you can do this with : sudo apt-get install python3-pip and then run it with: sudo pip3 install (package name).**

Thanks for Reading!
