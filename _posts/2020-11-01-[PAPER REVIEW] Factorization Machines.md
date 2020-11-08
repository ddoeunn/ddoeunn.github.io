---
layout: post
title: PAPER REVIEW_Factorization Machines
tags: [paper review, recommender system, matrix factorization, implicit feedback, Bayesian]
use_math: true
---

---
# **1. Prediction Under Sparsity**


---
# **2. Factorization Machines(FM)**
## 2.1 Model
#### Model Equation (degree $d=2$) :  
$$
\hat{y}(\mathbf{x}) := w_0 + \underset{i=1}{\sum^{n}}w_i x_i +
\underset{i=1}{\sum^{n}}\underset{j=i+1}{\sum^{n}} \left\langle \mathbf{v}_i, \mathbf{v}_j \right\rangle x_i x_j
$$

$$
\left\langle \mathbf{v}_i, \mathbf{v}_j \right\rangle =
\underset{f=1}{\sum^{k}}v_{i, f} \cdot v_{j, f}\\
$$

where $k \in \mathrm{N}_0^+$  is the hyperparameter that defines the dimensionality of the factorization.

#### Model Parameters :  
$$
w_0 \in \mathbb{R}, \quad \mathbf{w} \in \mathbb{R}^n, \quad \mathbf{V} \in \mathbb{R}^{n \times k}
$$

row  $\mathbf{v}_i$ within $\mathbf{V}$ describes the $i$-th variable with $k$ factors.  

2-way FM captures all single and pairwise interactions between variables.  
* $w_0$ : global bias
* $w_i$ : models the strength of the $i$-th variable
* $\hat{w}_{i, j} := \left\langle \mathbf{v}_i, \mathbf{v}_j \right\rangle$
