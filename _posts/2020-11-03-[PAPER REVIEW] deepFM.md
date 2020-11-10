---
layout: post
title: PAPER REVIEW-DeepFM A Factorization Machine based Neural Network for CTR Prediction
tags: [paper review, recommender system, factorization machine, Neural Network]
use_math: true
---

***Guo, Huifeng, et al. 2017 "DeepFM: a factorization-machine based neural network for CTR prediction."***

This [paper](https://arxiv.org/pdf/1703.04247.pdf) propose new model 'DeepFM' which combines the power of Factorization Machine for recommendation and the power of deep learning for feature learning in a neural network architecture. Unlike FM which learns only low-order feature interactions, DeepFM is possible to learn both low- and high- order feature interactions. Thus it can learn more sophisticated feature interactions behind user behaviors in CTR for recommender systems. Also, compared to '[Wide&Deep](https://dl.acm.org/doi/pdf/10.1145/2988450.2988454?casa_token=4hAHS_oEAZ8AAAAA:cpLXf11xBllDhZeLIMEREVh_ZwEWV8TsSqd8wdexklndOB3Hd9ZuNvSajbVa01Py2TYciz8VUmKsHy4)' model DeepFM has a shared input to its 'wide' and 'deep' part, with no need of feature engineering besides raw features. 


---
# **1. Problem Setup and Notation**  
The task of CTR prediction is to build a prediction model $\hat{y} = $ CTR_model$(x)$ to estimate the probability of a user clicking.

Suppose the data set for training consists of $n$ instances $(\mathcal{X}, y)$ where :  

$$
\begin{align*}
y &= \begin{cases}
1\quad \text{the user clicked item}\\
0\quad \text{otherwise}\\
\end{cases}\\
\mathcal{X} &= m \text{-fields data record}\\
\end{align*}
$$

Each instance is converted to $(x, y)$ where :

$$
x = \left[ x_{field_1}, x_{field_2}, \cdots, x_{field_m} \right]_{1 \times d} \\
x_{field_i} \in \mathbb{R}^{1 \times m_i},\quad \underset{i=1}{\sum^{m}}m_i = d
$$

<p align="center">
with $x_{field_i}$ being the vector representation of the $i$-th field of $\mathcal{X}$
</p>

---
# **2. DeepFM**
DeepFM consists of 2 components, FM(learn low-order feature interaction) and deep(learn high-order feature interaction) that share the same input. It brings 2 benefits :
* it learns both low- and high- order feature interactions from raw features
* no need for expertise feature engineering of the input

#### Input
Because the raw featrue input vector for CTR prediction is usually highly sparse, super high-dimensional, categorical-continuous-ised, and grouped in fields(e.g. gender, location, age), an embedding layer is suggested to compress the input vector to a low-dimensional$(k)$, dense real-value vector.

* the output of the embedding layer :

$$
a^{(0)} = [e_1, e_2, \cdots, e_m]
$$

$$
\begin{cases}
e_i &= \text{ the embedding of }i\text{-th field} \in \mathbb{R}^{1 \times k}\\
m &= \text{ the number of fields}\\
\end{cases}
$$

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/deepfm_figure4.PNG?raw=true" width=350 alter="architecture of FM ">
</p>


---
#### Combined prediction model

$$
\hat{y} = sigmoid(y_{FM} + y_{DNN})
$$

<p align="center">
where $y_{FM}$ is the output of FM component and $y_{DNN}$ is the output of deep component.
</p>

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/deepfm_figure1.PNG?raw=true" width=350 alter="architecture of FM ">
</p>




---
### 2.1 FM Component
See FM post [here](https://ddoeunn.github.io/2020/11/01/PAPER-REVIEW-Factorization-Machines.html).
The latent feature vectors $V_i$ in FM now server as network weights which are learned and used to compress the input field vectors to the embedding vectors.

$$
y_{FM} = \left\langle w, x \right\rangle + \underset{j_1=1}{\sum^d}\underset{j_2=j_1+1}{\sum^d}
\left\langle V_i, V_j  \right\rangle  x_{j_1}x_{j_2}
$$

<p align="center">
where $w \in \mathbb{R}^d$ ,$V_i \in \mathbb{R}^k$
</p>

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/deepfm_figure2.PNG?raw=true" width=300 alter="architecture of FM ">
</p>



---
### 2.2 Deep Component
The deep component is a feed-forward neural network, which is used to learn high-order feature interactions.


* $a^{(0)}$ is fed into the deep neural network, and the forward process is :

$$
a^{(l+1)} = \sigma \left( W^{(l)}a^{(l)} + b^{(l)} \right)
$$

$$
\begin{cases}
l &= \text{layer depth}\\
\sigma &= \text{activation function}\\
a^{(l)} &= \text{output of the }l\text{-th layer}\\
W^{(l)} &= \text{model weight of the }l\text{-th layer}\\
b^{(l)} &= \text{bias of the }l\text{-th layer}
\end{cases}
$$


* a dense real-value feature vector is generated, which is finally fed into the sigmoid function for CTR prediction :

$$
y_{DNN} = \sigma \left( W^{\lvert H \rvert + 1} \cdot a^{H} + b^{\lvert H \rvert + 1} \right)
$$

<p align="center">
where $\lvert H \rvert = $the number of hidden layers
</p>

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/deepfm_figure3.PNG?raw=true" width=350 alter="architecture of FM ">
</p>



---
# **3. Experiments**
It compared DeepFM with Logistic Regression(LR), Factorization Machines(FM), FNN, PNN and Wide&Deep(LR&DNN, FM&DNN) using AUC and logloss as evaluation metric.  

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/deepfm_table1.PNG?raw=true" width=350 alter="architecture of FM ">
</p>

#### Result
* Learning feature interactions improves the performance of CTR prediction model.  
; LR performs worse than the other models
* Learning high- and low-order feature interactions simultaneously and properly improves the performance.
; DeepFM outperforms the models that learn only low-order feature interactions(FM) or only high-order feature interactions(FNN, PNN)
* Learning high- and low-order feature interactions simultaneously while sharing the same feature embedding improves the performance.  
; DeepFM outperforms the models that learn high- and low-order feature interactions using separate feature embeddings(LR&DNN, FM&DNN).

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/deepfm_table2.PNG?raw=true" width=350 alter="architecture of FM ">
</p>


---
# **4. Summary**
* DeepFM trains a deep component and an FM component jointly that share the same input.
* no need any pre-training
* learns both high- and low- feature interactions
* no need for expertise feature enginnering of the input as required in Wide&Deep


---
### Reference
[[1]](https://arxiv.org/pdf/1703.04247.pdf)Guo, Huifeng, et al. "DeepFM: a factorization-machine based neural network for CTR prediction." arXiv preprint arXiv:1703.04247 (2017).  
[[2]](https://dl.acm.org/doi/pdf/10.1145/2988450.2988454?casa_token=4hAHS_oEAZ8AAAAA:cpLXf11xBllDhZeLIMEREVh_ZwEWV8TsSqd8wdexklndOB3Hd9ZuNvSajbVa01Py2TYciz8VUmKsHy4)Cheng, Heng-Tze, et al. "Wide & deep learning for recommender systems." Proceedings of the 1st workshop on deep learning for recommender systems. 2016.  
[[3]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5694074&casa_token=VLizdwKkgvYAAAAA:RfhMXZDECy3BjXxZShRfTW4SAYl2WhQqYsqNGCemlbVu09E7S7NVT2SjM8qdZCHSJszLjrwASTU&tag=1)Rendle, Steffen. "Factorization machines." 2010 IEEE International Conference on Data Mining. IEEE, 2010.  
