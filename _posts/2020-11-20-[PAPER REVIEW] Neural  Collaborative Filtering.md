---
layout: post
title: PAPER REVIEW-Neural Collaborative Filtering
tags: [paper review, recommender system, matrix factorization, implicit feedback, Deep Learning]
use_math: true
---

***He, Xiangnan, et al. "Neural collaborative filtering." Proceedings of the 26th international conference on world wide web. 2017.***



This [paper](https://dl.acm.org/doi/pdf/10.1145/3038912.3052569?casa_token=8SoRxfF0gD8AAAAA:bb8-vwZJGm7ChbPo2R-cevA5h39gnYZNXjOeX4_wcl2FOxSCicp83cv-qxzyaJDJeZGMWQLNpQlcJk0) present a general framework named NCF, short for Neural network-based Collaborative Filtering, by replacing the inner product with a neural architecture that can learn an arbitrary function from data.  

It points out that despite the effectiveness of Matrix Factorization for collaborative filtering, its performance can be hindered by the simple choice of the interaction function — inner product which simply combines the multiplication of latent features linearly.
And it shows significant improvements of proposed NCF framework over the state-of-the-art methods by experiments on two real-world datasets(movielens, pinterest).



---
# **1. Learning from Implicit Data**

<p align="center">
$\text{Let }M = \text{the number of users}$, $N = \text{the number of items}$
</p>


* Define the user-item interaction matrix from users' implicit feedback as :  

$$
\mathbf{Y} = [y_{ui}] \in \mathbb{R}^{M \times N}\\
y_{ui} = \begin{cases}
1, \quad \mbox{if interaction (user u and item i) is observed}\\
0, \quad \mbox{o.w}
\end{cases}
$$

* Prediction model

$$
\hat{y}_{ui} = f(u, i \vert \Theta)
$$

 where $\Theta$ denotes model parameters and $f$ denotes the function that maps model parameter to the predicted score(termed as an interaction function in this paper).


---
# **2. Matrix Factorization**
MF can be deemed as a linear model of latent factors. See FM post [here](https://ddoeunn.github.io/2020/11/01/PAPER-REVIEW-Factorization-Machines.html).  
* Prediction model

$$
\text{Let } \mathbf{p}_u = \text{the latent vector for user }u, \mathbf{q}_i = \text{the latent vector for item }i\\
\hat{y}_{ui} = f(u, i \vert \mathbf{p}_u, \mathbf{q}_i) = \mathbf{p}^T_u \mathbf{q}_i = \underset{k=1}{\sum^K}p_{uk}q_{ik}\\
\text{where }K=\text{the dimension of the latent space}
$$

* How the inner product function can limit the expressiveness of MF?  







---
# **3. Neural Collaborative Filtering**


## **3.1 General Framework**





## **3.2 Generalized Matrix Factorization**





## **3.3 Multi-Layer Perceptron**



## **3.4 Fusion of GMF and MLP**





---
# **4. Experiments**
