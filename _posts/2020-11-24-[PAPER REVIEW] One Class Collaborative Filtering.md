---
layout: post
title: PAPER REVIEW-One Class Collaborative Filtering
tags: [paper review, recommender system, matrix factorization, implicit feedback]
use_math: true
---

***Pan, Rong, et al. "One-class collaborative filtering." 2008 Eighth IEEE International Conference on Data Mining. IEEE, 2008.***

In this [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4781145&casa_token=gAwmW-BRz0kAAAAA:FMwzCG7LF_0FMoZs1AUWtPxDyHZhEhYJxIG57WVUak5zKQzLC1d4i9MIv-YdV0YbSx_S5_FoA0A&tag=1) collaborative filtering with only positive examples is referred as One Class Collaborative Filtering(OCCF).

In OCCF problem, the training data usually consist simply of binary data reflecting a user's action or inaction and has two characteristics; 1. extremely sparse data(a small fraction are positive examples) 2. negative examples and unlabeled positive examples are mixed together(unable to distinguish them).

This paper proposes 2 frameworks to tackle OCCF.  
*First based on weighted low rank approximation, second based on negative example sampling.*


---
# **1. Implicit Feedback**
More often than in situations that users express ratings explicitly such as a 1-5 scale in Netflix, the ratings can be implicitly expressed by users' behavior such as click or not-click and bookmark or not-bookmark.  
Advantage and disadvantages follow :   
* more common and easier to obtain
* extremely sparse data(a small fraction are positive examples)
* hard to identify representative negative examples  
(unable to distinguish negative examples and unlabeled positive examples)


---
# **2. Problem Definition**  

$$
m = \text{the number of users}\\
n = \text{the number of items}\\
\mathbf{R} = [r_{ij}]_{m\times n} = \text{previous viewing information matrix}\\
r_{ij} = \begin{cases}
1 \quad \mbox{positive example} \\
0 \quad \mbox{unknown(missing)}
\end{cases}
$$

<p align="center">
Our task is to identify potential positive examples from the missing data bases on $\mathbf{R}$
</p>

---
# **3. Weighting based Approach**
#### Idea  
: to give different weights to the error terms of positive examples and negative examples in the objective function  

#### Weight matrix  

$$
\begin{align*}
\mathbf{W} &= [w_{ij}]_{m\times n} \in \mathbb{R}_{+}^{m\times n}\\
w_{ij}  &= \begin{cases}
 1 \qquad \qquad \quad  \mbox{        if } r_{ij} = 1\\
0 \le w_{ij} \le 1 \quad \mbox{if } r_{ij} = 0
\end{cases}
\end{align*}
$$

---
#### Weighting schemes
<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/wrmf_table1.PNG?raw=true"  alt="table1"  width="400">
</p>

1. assumes that a missing data being a negative example has an equal chance over all users or all items  
$\rightarrow$ uniformly assign a weight $\delta \in [0, 1]$ for negative examples

2. assumes that if a user has more positive examples, it is more likely that she does not like the other items  
$\rightarrow$ the missing data for this user is negative with higher probability

3. assumes that if an item has fewer positive examples, the missing data for this item is negative with higher prob.

---
#### Low rank matrix

$$
\mathbf{X} = \mathbf{UV}^T \in \mathbb{R}^{m \times n}\\
\mathbf{U} = \begin{bmatrix}
\mathbf{u}_1\\
\vdots\\
\mathbf{u}_{m}
\end{bmatrix} \in \mathbb{R}^{m \times d} \quad
\mathbf{V} = \begin{bmatrix}
\mathbf{v}_1\\
\vdots\\
\mathbf{v}_{n}
\end{bmatrix} \in \mathbb{R}^{n \times d}
$$

---
#### Objective function

$$
\begin{align*}
L(\mathbf{U, V}) &= \frac{1}{2}\underset{i, j}{\sum}w_{ij}\left(r_{ij} - \mathbf{u}_i \mathbf{v}_j^T\right)^2 + \lambda \left(\lVert \mathbf{U} \rVert^2 + \lVert \mathbf{V} \rVert^2  \right)\\
&=\frac{1}{2}\underset{i, j}{\sum}w_{ij} \left(\left(r_{ij} - \mathbf{u}_i
\mathbf{v}_j^T\right)^2  +\lambda \left( \lVert \mathbf{u}_i \rVert^2 + \lVert \mathbf{v}_j \rVert^2 \right) \right)
\end{align*}
$$

---
#### Partial derivatives

$$
\frac{\partial L(\mathbf{U, V})}{\partial u_{ik}}
= -\underset{j}{\sum}w_{ij}\left( r_{ij} - \mathbf{u}_i\mathbf{v}^T_j \right)v_{jk} + \lambda\left( \underset{j}{\sum}w_{ij} \right)u_{ik}
$$

$$
\begin{align*}
\frac{\partial L(\mathbf{U, V})}{\partial \mathbf{u}_i}
&=\left( \frac{\partial L(\mathbf{U, V})}{\partial u_{i1}}, \cdots,
\frac{\partial L(\mathbf{U, V})}{\partial u_{id}} \right)\\
&= \mathbf{u}_i\left(\mathbf{V}^T \tilde{\mathbf{W}_i}\mathbf{V} + \lambda \left( \underset{j}{\sum }w_{ij} \right) \mathbf{I}  \right) - \mathbf{r}_i \tilde{\mathbf{W}_i}\mathbf{V}
\end{align*}
$$

$$
\text{where } \tilde{\mathbf{W}_i} = diag(\mathbf{w}_i) \in \mathbf{R}^{n \times n}, \mathbf{w}_i = [w_{i1}, \cdots, w_{in}]
$$

---
#### weighted ALS (update $\mathbf{u}_{i}$)
<p align="center">
Fixing $\mathbf{V}$ and solving $\frac{\partial L(\mathbf{U, V})}{\partial \mathbf{u}_i} = 0$, and update $\mathbf{u}_i$ by
</p>

$$
\mathbf{u}_i = \mathbf{r}_i \tilde{\mathbf{W}_i}\mathbf{V}\left(\mathbf{V}^T \tilde{\mathbf{W}_i}\mathbf{V} + \lambda \left( \underset{j}{\sum }w_{ij} \right) \mathbf{I}  \right)^{-1} \quad \forall 1 \le i \le m
$$

---
#### weighted ALS (update $\mathbf{v}_{j}$)
<p align="center">
Fixing $\mathbf{U}$ and solving $\frac{\partial L(\mathbf{U, V})}{\partial \mathbf{v}_j} = 0$, and update $\mathbf{v}_j$ by
</p>

$$
\mathbf{v}_j = \mathbf{r}_j \tilde{\mathbf{W}_j}\mathbf{U}\left(\mathbf{U}^T \tilde{\mathbf{W}_j}\mathbf{U} + \lambda \left( \underset{i}{\sum }w_{ij} \right) \mathbf{I}  \right)^{-1} \quad \forall 1 \le j \le n
$$

$$
\text{where } \tilde{\mathbf{W}_j} = diag(\mathbf{w}_j) \in \mathbf{R}^{m \times m}, \mathbf{w}_j = \begin{pmatrix}
w_{1j}\\
\vdots\\
w_{mj}
\end{pmatrix}
$$


<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/wrmf_algorithm1.PNG?raw=true"  alt="algorithm"  width="400">
</p>


---
# **4. Sampling-based Approach**
Because of too many  negative examples compared to positive ones, it is costly and not necessary to learn the model on all entries of $\mathbf{R}$.

#### Idea  
: to sample some missing values as negative examples based on some sampling strategies  

---
#### Notations

$$
\hat{\mathbf{P}} = \text{sampling probability matrix}\\
q = \text{negative sample size}
$$

---
#### Sampling schemes

  + Uniform random sampling  
<p align="center">$\hat{p}_{ij} \propto 1$</p>  
all the missing data share the same probability of being sampled as negative examples.
  + User-Oriented sampling  
<p align="center">$\hat{p}_{ij} \propto \underset{i}{\sum}I(r_{ij}=1)$</p>  
if a user hav viewed more items, those items that she has not viewed could be nagative with higher probability.
  + Item-Oriented sampling  
<p align="center">$\hat{p}_{ij} \propto 1/\underset{j}{\sum}I(r_{ij}=1)$</p>  
if an item is viewed by less users, those users that have not viewed the item will not view it either

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/wrmf_algorithm2.PNG?raw=true"  alt="algorithm"  width="400">
</p>

---
## Reference
[[1]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4781145&casa_token=gAwmW-BRz0kAAAAA:FMwzCG7LF_0FMoZs1AUWtPxDyHZhEhYJxIG57WVUak5zKQzLC1d4i9MIv-YdV0YbSx_S5_FoA0A&tag=1) Pan, Rong, et al. "One-class collaborative filtering." 2008 Eighth IEEE International Conference on Data Mining. IEEE, 2008.
