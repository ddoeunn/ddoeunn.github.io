---
layout: post
title: PAPER REVIEW-Factorization Machines
tags: [paper review, recommender system, matrix factorization, factorization machine]
use_math: true
---

***Rendle, Steffen. 2010 "Factorization machines."***  

This [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5694074&casa_token=iTXf467ebBkAAAAA:ucq_oZrtnr-1UuHTjZ9LrRwraw9iOjXmaAMWuBqTrHEAzDMxao1Mv-TTqf7m0flmNdpUppByN4Q) introduces Factorization Machines(FM) which are a general predictor working with any real valued feature vector.(In contrast to this, other factorization models work only on very restricted input data.)  FMs model all interaction between variables using factorized parameters. Thus they are able to estimate interactions even in problems with huge sparsity like recommender systems. Also, FMs have advantage that the model equation of FMs can be calculated in linear time and thus FMs can be optimized directly.  

*Implementation of FM using Tensorflow -> [here!](https://github.com/ddoeunn/recommender-system-implementation)*

---
# **1. Problem Set up and Notations**
Estimate a function $y : \mathbb{R}^n \rightarrow T$ from  a real valued feature vector $\mathbf{x} \in \mathbb{R}^n$ to a target domain $T$.

$$
\begin{align*}
D &= \{(\mathbf{x}^{(1)}, y^{(1)}), (\mathbf{x}^{(2)}, y^{(2)}), \cdots \} : \text{training dataset}\\
m(\mathbf{x}) &= \text{the number of non-zero elements in } \mathbf{x}\\
\bar{m}_D &= \text{the average of non-zero elements }m(\mathbf{x}) \text{ of all vectors } \mathbf{x} \in D
\end{align*}
$$

---
# **2. Factorization Machines(FM)**
### 2.1 Model
#### Model Equation (degree $d=2$) :  

$$
\hat{y}(\mathbf{x}) := w_0 + \underset{i=1}{\sum^{n}}w_i x_i +
\underset{i=1}{\sum^{n}}\underset{j=i+1}{\sum^{n}} \left\langle \mathbf{v}_i, \mathbf{v}_j \right\rangle x_i x_j
$$

$$
\left\langle \mathbf{v}_i, \mathbf{v}_j \right\rangle =
\underset{f=1}{\sum^{k}}v_{i, f} \cdot v_{j, f}\\
$$

<p align="center">
where $k \in \mathrm{N}_0^+$  is the hyperparameter that defines the dimensionality of the factorization.
</p>

---
#### Model Parameters :  

$$
w_0 \in \mathbb{R}, \quad \mathbf{w} \in \mathbb{R}^n, \quad \mathbf{V} \in \mathbb{R}^{n \times k}
$$

<p align="center">
row  $\mathbf{v}_i$ within $\mathbf{V}$ describes the $i$-th variable with $k$ factors.  
</p>

2-way FM captures all single and pairwise interactions between variables.  
* $w_0$ : global bias
* $w_i$ : models the strength of the $i$-th variable
* $\hat{w}_{i, j} := \left\langle \mathbf{v}_i, \mathbf{v}_j \right\rangle$ : models the interaction between the $i$th and $j$th by factorizing it.

---
#### Parameter Estimation Under Sparsity :  
In sparse settings, there is usually not enough data to estimate interactions between variables directly and independently. FM can estimate interactions even in these settings well because they break the independence of the interaction parameters by factorizing them. In general this means that the data for one interaction helps also to estimate the parameters for related interactions.

---
#### Computation :  
* Complexity of straight forward computation of model eq. is in $O(kn^2)$, because all pairwise interactions have to be computed.
* But with reformulating it drops to linear runtime $O(kn)$.  

$$
\begin{align*}
\underset{i=1}{\sum^{n}}\underset{j=i+1}{\sum^{n}} \left\langle  \mathbf{v}_i, \mathbf{v}_j \right\rangle x_i x_j
&= \frac{1}{2} \underset{i=1}{\sum^{n}}\underset{j=1}{\sum^{n}} \left\langle  \mathbf{v}_i, \mathbf{v}_j \right\rangle x_i x_j - \frac{1}{2}\underset{i=1}{\sum^{n}} \left\langle  \mathbf{v}_i, \mathbf{v}_j \right\rangle x_i x_i\\
&= \frac{1}{2} \left(\underset{i=1}{\sum^{n}} \underset{j=1}{\sum^{n}}\underset{f=1}{\sum^{k}} v_{if}v_{jf}x_i x_j - \underset{i=1}{\sum^{n}}\underset{f=1}{\sum^{k}} v_{if}^2x_i^2 \right)\\
&= \frac{1}{2}\underset{f=1}{\sum^{k}} \left( \underset{i=1}{\sum^{n}} \underset{j=1}{\sum^{n}} v_{if} v_{jf} x_i x_j - \underset{i=1}{\sum^{n}} v_{if}^2x_i^2 \right)\\
&=\frac{1}{2}\underset{f=1}{\sum^{k}} \left( \left( \underset{i=1}{\sum^{n}}v_{if}x_i \right)  \left( \underset{j=1}{\sum^{n}}v_{jf}x_j \right) - \underset{i=1}{\sum^{n}}v_{if}^2 x_i^2 \right)\\
&=\frac{1}{2} \underset{f=1}{\sum^{k}} \left( \left( \underset{i=1}{\sum^{n}}v_{if} x_i \right)^2 - \underset{i=1}{\sum^{n}}v_{if}^2 x_i^2  \right)
\end{align*}
$$

* Because most of the elements in $\mathbf{x}$ are 0 under sparsity, computation of FM is in $O(k \bar{m}_D)$  

---
### 2.2 FM as Predictors

$$
y : \mathbb{R}^n \rightarrow T
$$

* Regression: $\hat{y}(\mathbf{x}), T=\mathbb{R}$
* Binary classification: the sign of $\hat{y}(\mathbf{x}), T = \{+, -\}$
* Ranking: the vectors $\mathbf{x}$ are ordered by the score of $\hat{y}(\mathbf{x}), T=\mathbb{R}$

---
### 2.3 Learning FM
The model paramters $w_0, \mathbf{w}, \mathbf{V}$ can be learned efficiently by gradient descent methods.  
Gradient of the FM model is :

$$
\begin{align*}
\frac{\partial}{\partial w_0} \hat{y}(\mathbf{x}) &= 1 \\
\frac{\partial}{\partial w_i} \hat{y}(\mathbf{x}) &= x_i \\
\frac{\partial}{\partial v_{i, f}} \hat{y}(\mathbf{x}) &= x_i \sum_{j=1}^{n}v_{j, f}x_j - v_{i, f}x_i^2
\end{align*}
$$

---
### 2.4 d-way FM

$$
\hat{y}(x) = w_0 + \sum_{i=1}^{n}w_i x_i +
\sum_{l=2}^{d} \sum_{i_1 = 1}^{n} \cdots \sum_{i_{l-1}}^{n}
\left( \prod_{j=1}^{l} x_{i_j} \right)
\left( \sum_{f=1}^{k_l} \prod_{j=1}^{l} v_{i_j, f}^{(l)}  \right)
$$

where the interaction parameters for the $l$-the interaction are factorized by the [PARAFAC](https://www.psychology.uwo.ca/faculty/harshman/wpppfac0.pdf) model with the model parameters:

$$
\mathbf{V}^{(l)} \in \mathbb{R}^{n \times k_l}, \quad k_l \in \mathbb{N}_0^+
$$

---
# **Summary**
#### Advantages of FM:  
* FMs allow parameter estimation under very sparse data where SVMs fail.
* FMs have linear complexity, can be optimized in the primal
* FMs are a general predictor that can work with any real valued feature vector.


---
### Reference
[[1]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5694074&casa_token=iTXf467ebBkAAAAA:ucq_oZrtnr-1UuHTjZ9LrRwraw9iOjXmaAMWuBqTrHEAzDMxao1Mv-TTqf7m0flmNdpUppByN4Q) Rendle, Steffen. "Factorization machines." 2010 IEEE International Conference on Data Mining. IEEE, 2010.
