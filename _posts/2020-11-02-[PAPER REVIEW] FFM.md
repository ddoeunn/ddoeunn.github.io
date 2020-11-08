---
layout: post
title: PAPER REVIEW_Field-aware Factorization Machines for CTR prediction
tags: [paper review, recommender system, matrix factorization, factorization machine, CTR]
use_math: true
---

Click-through rate(CTR) prediction plays an important role in computational advertising. This [paper](https://dl.acm.org/doi/pdf/10.1145/2959100.2959134?casa_token=2HFKgPvmUnQAAAAA:74DUN0wTfUgZu92OPlmGQsIpTlPVqJv7Dzjspa_ZMVJZ-k5j4e-Cw7hPzKusLJNY30O7VG8TXvcXCgI) establishes Field-aware Factorization Machines(FFM) as an effective method for classifying large sparse data including CTR prediction. FFM is a variant of [FM](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5694074) that utilizes information that features can be grouped into field for most CTR dataset.

---
# **1. CTR Dataset**
For most CTR datasets, "features" can be grouped into "field". In
Table1: example, three features ESPN, Vogue, and NBC, belong to theeld Publisher, and the other three features Nike, Gucci, and Adidas, belong to the eld Advertiser. FFM is a variant of FM that utilizes this information.

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/ffm_table1.PNG?raw=true"  alt="ctr example"  width="400">
</p>

---
# **2. FM**
See FM post [here](https://ddoeunn.github.io/2020/11/01/PAPER-REVIEW-Factorization-Machines.html).

$$
\phi_{FM}(\mathbf{w}, \mathbf{x}) = \underset{j_1=1}{\sum^{n}}\underset{j_2 = j_1+1}{\sum^{n}}(\mathbf{w}_{j_1}\cdot
   \mathbf{w}_{j_2})
   x_{j_1}x_{j_2}
$$

$$
\mathbf{w}_{j_1}\cdot
\mathbf{w}_{j_2} = \underset{f=1}{\sum^k}
w_{j_1, f}w_{j_2, f}
$$

<p align="center">
where $k \in \mathrm{N}_0^+$  is the hyperparameter that defines the dimensionality of the factorization.
</p>

#### Example

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/ffm_example.PNG?raw=true"  alt="ctr example"  width="400">
</p>

$$
\phi_{FM}(\mathbf{w}, \mathbf{x}) = \mathbf{w}_{ESPN} \cdot
\mathbf{w}_{Nike} +
\mathbf{w}_{ESPN} \cdot
\mathbf{w}_{Male} +
\mathbf{w}_{Nike} \cdot
\mathbf{w}_{Male}
$$


---
# **3. FFM**

$$
\phi_{FFM}(\mathbf{w}, \mathbf{x}) = \underset{j_1=1}{\sum^n} \underset{j_2=j_1+1}{\sum^n} (\mathbf{w}_{j_1, f_2} \cdot
  \mathbf{w}_{j_2, f_1})
  x_{j_1}x_{j_2}
$$


<p align="center">
where $f_1$ and $f_2$  are respectively the fields of $j_1$ and $j_2$
</p>

In FFMs, each feature has several latent vectors. Depending on the field of other features, one of them is used to do the inner product

#### Example

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/ffm_example.PNG?raw=true"  alt="ctr example"  width="400">
</p>

$$
\phi_{FFM}(\mathbf{w}, \mathbf{x}) = \mathbf{w}_{ESPN, A} \cdot
\mathbf{w}_{Nike, P} +
\mathbf{w}_{ESPN, G} \cdot
\mathbf{w}_{Male, P} +
\mathbf{w}_{Nike, G} \cdot
\mathbf{w}_{Male, A}
$$

---
### 3.1 Solving the Optimization Problem
#### sub-gradient
