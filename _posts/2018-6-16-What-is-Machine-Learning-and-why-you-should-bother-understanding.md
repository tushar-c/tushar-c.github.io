---
layout: post
published: true
title: What is Machine Learning And Why You Should Bother Understanding
---
![ML Picture]({{site.baseurl}}/images/1_2UjSSQwW0bns1lPIuRxccQ.png)

Welcome to the first post! Let us dive right in and try to answer the question the title asks : What is machine learning and why is it such a big deal, and moreover, why should we (**specifically those of us in biology and medicine, and people from all sorts of backgrounds too**) have knowledge of this technology? A valid but not very long-thought-on answer would be - because AI is going to change medicine and biology forever! 


That might be very well true, and the answer you are most likely to get in casual conversations. The deeper fact, and a better answer, is that while AI (of which machine learning is a subset) might not be our answer to all diseases, disorders and their respective cures, nevertheless, it is an extremely powerful tool to have as a medical practitioner. Machine Learning and other statistical methods operate not on hand written code, but rather on data - and sometimes huge chunks of data. Hospitals and research labs are generating huge amounts of data through patient diagnoses, RNA/DNA sequencing and other techniques.


The scenario where the need for machine learning becomes apparent is when we start asking questions like - "Are there correlations between disease X and symptoms Y? And if there are, are these correlations alone, or combined with other factors, good predictors of disease X?". You can see how the need for a lot of data becomes obvious in both questions. To answer the first question, we need data. To answer the second, we need suitable methods for processing that data. 


The methods we use for dealing with data and extracting insight are formally called 'Algorithms'. The term describes a set of computational steps that are executed to map (_convert_) an input to an output. Before we go on to other things, we have to make a key distinction here. 

While traditional algorithms (**such as sorting numbers, searching for a specific entry, and many more**) act through 'hard-coded' rules (**i.e. , if event X is true, do action Y, else do action Z**), machine learning differs in that the algorithms 'learn' the rules by themselves, we just provide the data. 

This is a key point, and one that often causes images of robots becoming our gods occur in our mind. To attempt an informal one line explanation to this 'learning' (**better ones will be presented in further posts**), the algorithms are penalized for making mistakes and rewarded (**or not penalized at all**) for being correct. This penalization occurs formally through a well defined error function (**do not be scared if you don't know what this means, we will understand all of this in time**), which the algorithm aims to minimize (**make as low as possible as we train for longer periods of time**).

An example is shown in the following figure: 
![error]({{site.baseurl}}/images/gradient_descent_error_by_iteration.png)


In this way, if we have medical data consisting of, say, scans of lungs, and we 'tag' (add an entry to) each image with whether some disease is present or not, then we can use these images to train an algorithm to recognize the presence or absence of that particular disease. Of course, this is just one example of where ML (Machine Learning) in medicine helps. 

A lot of biological applications exist too, such as understanding genetic codes and trying to predict whether these lead to some genetic condition or not (**a much harder task**). We will see many more of these examples further, but for now let's stop here on this subject. 

The above should convince you that knowing this technology is worth your time as the data generated will only keep going up, and the need for methods will also go up. To diagnose and treat better, we will need doctors and biologists who know how these algorithms work. At the same time, some knowledge of applying these algorithms is going to be useful too. 

This is where the motivation for our philosophy comes in, not only will we try to understand the math underlying these algorithms (**do not let your mind scream ABORT and don't worry, we will go right down to the core of every concept; and while understanding technical details helps, remember that imagination and visualization are more important, so always keep the bigger picture in mind**), we will also implement these algorithms, in the Python programming language. It is an easy to learn, but very powerful language that is found everywhere in modern day science. 

Another field of AI that we will explore is called **Deep Learning** which involves making large neural networks, composed of small individual units called neurons, that architecturally seem to mimic the brain, but are not really aiming to simulate the brain. Though we will explore these later, here is an example picture. We will talk about each of these component parts and the final network in much more detail later.

![neuralnetwork]({{site.baseurl}}/images/neuralnetwork.jpeg)


Also, the reader might have noticed that we are using the terms AI, Machine Learning, Deep Learning interchangeably. Machine Learning is a subfield of AI, Deep Learning is a subfield of Machine Learning. Here is a nice visual representing this fact:

![Subsets Of AI]({{site.baseurl}}/images/Machine Learning.png)

Understanding the data, the algorithm, implementing it, and finally pointing out some limitations will be our learning trajectory in each post. How to get the software up and running will be explained in the next post. For now, I hope you are excited to atleast 'take a look at' what this fascinating technology is, and are eager about understanding it. I'll leave this one here, and see you next time. 

Thanks for reading!
