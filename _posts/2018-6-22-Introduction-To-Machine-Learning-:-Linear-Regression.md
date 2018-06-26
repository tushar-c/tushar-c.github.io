---
layout: post
mathjax: true
math: true
---


![Machine Learning]({{site.baseurl}}/images/LR.png)

![Linear Regression]({{site.baseurl}}/images/Predict_Regression.png)

\begin{align}
\frac{\partial O}{\partial a_\iota^t} &= \sum_c^C \frac{\partial O}{\partial s_c^t}\frac{\partial s_c^t}{\partial b_\iota^t}\frac{\partial b_\iota^t}{\partial a_\iota^t}\\
\frac{\partial O}{\partial a_\phi^t} &= \sum_c^C \frac{\partial O}{\partial s_c^t}\frac{\partial s_c^t}{\partial b_\phi^t}\frac{\partial b_\phi^t}{\partial a_\phi^t}\\
\frac{\partial O}{\partial a_c^t} &= \frac{\partial O}{\partial s_c^t}\frac{\partial s_c^t}{\partial a_c^t}\\
\frac{\partial O}{\partial a_w^t} &= \sum_c^C \frac{\partial O}{\partial b_c^t}\frac{\partial b_c^t}{\partial b_w^t}\frac{\partial b_w^t}{\partial a_w^t}\\
\end{align}


**Coming Soon!!**

