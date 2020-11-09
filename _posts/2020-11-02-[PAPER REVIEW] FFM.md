---
layout: post
title: PAPER REVIEW_Field-aware Factorization Machines for CTR prediction
tags: [paper review, recommender system, matrix factorization, factorization machine, CTR]
use_math: true
---

Click-through rate(CTR) prediction plays an important role in computational advertising. This [paper](https://dl.acm.org/doi/pdf/10.1145/2959100.2959134?casa_token=2HFKgPvmUnQAAAAA:74DUN0wTfUgZu92OPlmGQsIpTlPVqJv7Dzjspa_ZMVJZ-k5j4e-Cw7hPzKusLJNY30O7VG8TXvcXCgI) establishes Field-aware Factorization Machines(FFM) as an effective method for classifying large sparse data including CTR prediction. FFM is a variant of [FM](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5694074) that utilizes information that features can be grouped into field for most CTR dataset.  
It compared FFM with Logistic regression(LM), Polynomial regression(Poly2) and FM using logloss as evaluation metric. FFM outperform other models in terms of logloss but also requires longer training time that LM and FM.  

---
# **1. CTR Dataset**
For most CTR datasets, "features" can be grouped into "field". In
Table1: example, three features ESPN, Vogue, and NBC, belong to the field Publisher, and the other three features Nike, Gucci, and Adidas, belong to the eld Advertiser. FFM is a variant of FM that utilizes this information.

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


### 3.1 Solving the Optimization Problem
* Use stochastic gradient methods and AdaGrad(adaptive learning rate).
* The initial values of $\mathbf{w}$ are randomly sampled from a uniform distribution between $[0, \frac{1}{\sqrt{k}}]$.
* The initial values of $G$ (accumulated sum of squared gradient for each coordinate $d=1, \cdots, k$) are set to one in order to prevent a large value of $(G_{j_1, f_2})^{-\frac{1}{2}}\_d$.
* At each step of SG a data point $(y, \mathbf{x})$ is sampled for updating $\mathbf{w}_{j_1, f_2}$ and $\mathbf{w}\_{j_2, f_1}$.
* Beacause $\mathbf{x}$ is highly sparse in this application, only update dimensions with non-zero values.

---
#### Optimization Problem

$$
\underset{\mathbf{w}}{min} \frac{\lambda}{2} \lVert \mathbf{w} \rVert^2_2 + \underset{i=1}{\sum^m}\log (1+\exp (-y_i \phi_{FFM}(\mathbf{w}, \mathbf{x})))
$$

---
#### Sub-gradient

$$
\mathbf{g}_{j_1, f_2} \equiv
\nabla_{\mathbf{w}_{j_1, f_2}}f(\mathbf{w}) = \lambda \cdot
\mathbf{w}_{j_1, f_2} +
\kappa \cdot \mathbf{w}_{j_2, f_1}
x_{j_1} x_{j_2}\\
\mathbf{g}_{j_2, f_1} \equiv
\nabla_{\mathbf{w}_{j_2, f_1}}f(\mathbf{w}) = \lambda \cdot
\mathbf{w}_{j_2, f_1} +
\kappa \cdot \mathbf{w}_{j_1, f_2}
x_{j_1} x_{j_2}
$$

<p align="center">
where $\kappa = \frac{\partial \log (1+\exp (-y \phi_{FFM}(\mathbf{w}, \mathbf{x})))}{\partial \phi_{FFM}(\mathbf{w, x})} = \frac{-y}{1+\exp (y \phi_{FFM}(\mathbf{w}, \mathbf{x}))}$
</p>

---
#### Accumulate the sum of squared gradient
for each coordinate $d=1, \cdots, k$, the sum of squared gradient is accumulated:

$$
(G_{j_1, f_2})_d \leftarrow
(G_{j_1, f_2})_d +
(g_{j_1, f_2})^2_d\\
(G_{j_2, f_1})_d \leftarrow
(G_{j_2, f_1})_d +
(g_{j_2, f_1})^2_d
$$

---
#### Update Rule

$$
(w_{j_1, f_2})_d \leftarrow
(w_{j_1, f_2})_d -
\frac{\eta}{\sqrt{(G_{j_1, f_2})_d}}
(g_{j_1, f_2})_d\\
(w_{j_2, f_1})_d \leftarrow
(w_{j_2, f_1})_d -
\frac{\eta}{\sqrt{(G_{j_2, f_1})_d}}
(g_{j_2, f_1})_d
$$

<p align="center">
where $\eta$ is a user-specified learning rate.
</p>

---
#### Algorithm
<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/ffm_algorithm1.PNG?raw=true"  alt="algorithm"  width="400">
</p>


---
### 3.2 Adding Field Information
*  LIBSVM data format:
Each (feat, val) pair indicates feature index and value.

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/ffm_datatype1.PNG?raw=true"  alt="dataformat"  width="250">
</p>


* FFM data format:
Extend LIBSVM format, must assign the corresponding field to each feature.  

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/ffm_datatype2.PNG?raw=true"  alt="dataformat"  width="350">
</p>

---
#### Categorical Features
Apply same setting as LIBSVM that features with zero values are not stored. Every categorical feature is transformed to several binary ones. Consider each category as a field to add the field information.

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/ffm_example1.PNG?raw=true"  alt="example"  width="230">
</p>

The above example instance becomes :

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/ffm_example2.PNG?raw=true"  alt="example"  width="300">
</p>

---
#### Numerical features
There are 2 possible ways to assign field. A naive way is to treat each feature as a dummy filed, another way is to discretize each numerical feature to a categorical one.

* Example data
<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/ffm_example3.PNG?raw=true"  alt="example"  width="300">
</p>

*  treat each feature as a dummy field.

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/ffm_example4.PNG?raw=true"  alt="example"  width="350">
</p>

*  discretize each numerical feature to a categorical one and then use the same setting for categorical features to add field information

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/ffm_example5.PNG?raw=true"  alt="example"  width="250">
</p>


---
# **4. Experiment**
It compared FFM with Logistic regression(LM), Polynomial regression(Poly2) and FM on 2 CTR datasets using logloss as evaluation metric.  
It used "early stopping" which terminates the training process before reaching the best result on training data to avoid over-fitting, because find that unlike LM or Poly2, FFM is sensitive to the number of epochs.

#### Dataset
2 CTR dataset "Criteo" and "Avazu" from kaggle competitions. Because the labels in the test sets are not available, split the available data to training and validation.

#### Evaluation metric

$$
\text{logloss} = \frac{1}{m}\underset{i=1}{\sum^m}\log (1 + \exp (-y_i \phi(\mathbf{w}, \mathbf{x}_i)))
$$

<p align="center">
where $m$ is the number of test instances.
</p>

#### Results
See result table in [paper](https://dl.acm.org/doi/pdf/10.1145/2959100.2959134?casa_token=2HFKgPvmUnQAAAAA:74DUN0wTfUgZu92OPlmGQsIpTlPVqJv7Dzjspa_ZMVJZ-k5j4e-Cw7hPzKusLJNY30O7VG8TXvcXCgI) Table3.
* FFM outperform other models in terms of logloss
* But FFM also requires longer training time that LM and FM.
* In contrary, though the logloss of LM is worse than other models, it is significantly faster.
* There is a clear trade-off between logloss and speed.
* FM is a good balance between logloss and speed.


---
## Reference
[[1]](https://dl.acm.org/doi/pdf/10.1145/2959100.2959134?casa_token=jwaMxk647dAAAAAA:Vfm5UJYB8yDWqKtpayKx7FNERu2jF0TgY1tR8tKAcc7M-9FAAAtXlmZYYNOSgznR2Jm07wX1eCxz-bM) Juan, Yuchin, et al. "Field-aware factorization machines for CTR prediction." Proceedings of the 10th ACM Conference on Recommender Systems. 2016.  
[[2]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5694074) Rendle, Steffen. "Factorization machines." 2010 IEEE International Conference on Data Mining. IEEE, 2010.
