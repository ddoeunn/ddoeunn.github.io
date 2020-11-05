---
layout: post
title: PAPER REVIEW_Logistic Matrix Factorization for Implicit Feedback Data
tags: [paper review, recommender system, matrix factorization, implicit feedback]
use_math: true
---
This [paper](http://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf) presents Logistic Matrix Factorization, a new probabilistic model for matrix factorization with implicit feedback. The model has benefit that it can model the probability that a user will prefer a specific item.  It compared Logistic MF with [IMF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4781121)(Implicit Matrix Factorization) model using MPR evaluation metric and showed it to outperform IMF.  

---
# **1. Problem Setup and Notation**  
Assume that we have a set of non-negative feedback values associated with each pair of users and items in our domain.
* $U = \left( u_1, \cdots, u_n \right)$: a group of $n$ users
* $I = \left( i_1, \cdots, i_m \right)$: a group of $m$ items
* $\mathbf{R} = \left( r_{ui} \right)\_{n \times m}$: a user-item observation matrix ,where $r\_{ui} \in \mathbb{R}_{\ge 0}$ represents the number of times that user $u$ interacted with item $i$

For any entries $r_{ui}$ where user $u$ does not interact with item $i$, place 0's.  Note that a value 0 does not imply that the user does not prefer the item, but could simply imply that the user does not know about the item.
Then, our goal is to find the top recommended items for each user for each item that they have not yet interacted with.


---
# **2. Logistic MF**
### 2.1 Matrix Factorization
Factorizing the observation matrix $\mathbf{R}$ by 2 lower dimensional matrices $ \mathbf{X}\_{n \times f}$ and $\mathbf{Y}\_{m \times f}$ ,where $f=$ the number of latent factors.

$$ \mathbf{R} \approx \mathbf{X}_{n \times f}
\mathbf{Y}^T_{m \times f} $$

$$
\mathbf{X}_{n \times f} = \begin{pmatrix} \mathbf{x}_1  \\ \vdots\\ \mathbf{x}_n \end{pmatrix} ,\mathbf{Y}_{m \times f} = \begin{pmatrix} \mathbf{y}_1  \\ \vdots\\ \mathbf{y}_m \end{pmatrix},
\mathbf{x}_u^T = \begin{pmatrix} x_{u1}  \\ \vdots\\ x_{um} \end{pmatrix},
\mathbf{y}_i ^T= \begin{pmatrix} y_{i1}  \\ \vdots\\ y_{in} \end{pmatrix}
 $$

---
### 2.2 Define Random Variable
Let $l_{ui}$ be the event that user $u$ has chosen to interact with item $i$.

$$
l_{ui} =
\begin{cases}
1, r_{ui}\ne 0 \text{  represents positive  } \\
0, r_{ui}=0 \text{  represents negative }
\end{cases}
\sim Bernoulli(p(l_{ui}))
$$

Let the probability of this event occurring be distributed  according to a logistic function parameterized by the sum of the inner product of user and item latent factor vectors and biases.

$$
p(l_{ui}\vert \mathbf{x}_u, \mathbf{y}_i, \beta_u, \beta_i) = \frac{exp(\mathbf{x}_u\mathbf{y}_i^T+\beta_u + \beta_i)}{1+exp(\mathbf{x}_u\mathbf{y}_i^T+\beta_u + \beta_i)}\\
\begin{cases}
\mathbf{x}_u = \text{latent vector of user } u\\
\mathbf{y}_i = \text{latent vector of item } i\\
\beta_u = \text{user } u \text{ bias}\\
\beta_i = \text{item } i \text{ bias}\\
 \end{cases}
$$

---
### 2.3 Define Confidence  

$$  
c_{ui} = \alpha r_{ui}  =
\begin{cases}> 0 , r_{ui}\ne 0 \\
=0, r_{ui}=0\\
\end{cases}\\
\text{,where } \alpha\text{ is a tuning parameter}\\
\begin{cases}
\text{increasing } \alpha \Leftrightarrow \text{ more weight on the non-zero entries(positive feedback)}\\
\text{decreasing } \alpha \Leftrightarrow \text{ more weight on the zero entries(negative feedback)}\\
\end{cases}  
$$

---
### 2.4 Bayes Theorem  

$$
p(\mathbf{X}, \mathbf{Y},\beta_u, \beta_i \vert \mathbf{R})\propto p(\mathbf{R}\vert \mathbf{X}, \mathbf{Y},\beta_u, \beta_i)p(\mathbf{X})p(\mathbf{Y})
$$  


#### Likelihood
Assume that all entries of $\mathbf{R}$ are independent.  

$$
L(\mathbf{R}\vert \mathbf{X, Y}, \beta_u, \beta_i) =
\prod_{u, i}p(l_{ui}\vert\mathbf{x}_u, \mathbf{y}_i, \beta_u, \beta_i)^{\alpha r_{ui}} \times \left( 1-p(l_{ui}\vert\mathbf{x}_u, \mathbf{y}_i, \beta_u, \beta_i) \right)
$$
#### Prior Probability  

$$
p(\mathbf{X}\vert \sigma^2) = \prod_{u}N(\mathbf{x}_u \vert \mathbf{0}, \sigma_u^2\mathbf{I})
$$

####  Posterior Probability
Taking the log of posterior and replacing constant terms with a scaling parameter $\lambda$.  

$$
\log p(\mathbf{X}, \mathbf{Y},\beta_u, \beta_i \vert \mathbf{R})
\propto \log p(\mathbf{R}\vert \mathbf{X}, \mathbf{Y},\beta_u, \beta_i ) + \log p(\mathbf{X}) + \log p(\mathbf{Y})
$$

$$
\log p( \mathbf{X}, \mathbf{Y},\beta_u, \beta_i \vert \mathbf{R})\\
\propto \sum_{u, i} \{ \alpha r_{ui} \log p(l_{ui} \vert \mathbf{x}_u, \mathbf{y}_i, \beta_u, \beta_i)+\log \left ( 1- p(l_{ui} \vert \mathbf{x}_u, \mathbf{y}_i, \beta_u, \beta_i)\right)\}\\
+\sum_{u}\log N(\mathbf{x}_u \vert \mathbf{0}, \sigma^2_u\mathbf{I}) + \sum_{i}\log N(\mathbf{y}_i \vert \mathbf{0}, \sigma^2_i\mathbf{I})\\
= \sum_{u, i}\alpha r_{ui}(\mathbf{x}_u\mathbf{y}^T_i + \beta_u + \beta_i) - (1+\alpha r_{ui})\log (1+\exp (\mathbf{x}_u\mathbf{y}^T_i + \beta_u + \beta_i))\\
 -\frac{\lambda}{2} \lVert \mathbf{x}_u\rVert^2 - \frac{\lambda}{2} \lVert \mathbf{y}_i\rVert^2
$$  

---
# **3. Optimization**
Our goal is to learn $\mathbf{X}, \mathbf{Y}, \beta_u \beta_i$ that maximize the log posterior.

$$
\underset{\mathbf{X, Y}, \beta_u, \beta_i}{\operatorname{arg max}}\log p(\mathbf{X}, \mathbf{Y}, \beta_u, \beta_i \vert \mathbf{R})
$$

### ALS (Alternating Least Squares) Algorithm

1. Fix the user vectors $\mathbf{X}$ and user bias $\beta_u$ and take a step towards the gradient of the item vectors $\mathbf{Y}$ and item bias $\beta_i$
2. Fix the item vectors $\mathbf{Y}$ and item bias $\beta_i$ and take a step towards the gradient of the user vectors $\mathbf{X}$ and user bias $\beta_u$

#### Partial derivatives  
$$
\frac{\partial}{\partial \mathbf{y}_i} \log p(\mathbf{X}, \mathbf{Y}, \beta_u, \beta_i \vert \mathbf{R}) = \sum_{u}\alpha r_{ui}\mathbf{x}_u - \frac{(1+\alpha r_{ui})\mathbf{x}_u \exp (\mathbf{x}_u\mathbf{y}^T_i + \beta_u + \beta_i))}
{1+\exp (\mathbf{x}_u\mathbf{y}^T_i + \beta_u + \beta_i))} - \lambda \mathbf{y}_i\\
\frac{\partial}{\partial \mathbf{x}_u} \log p(\mathbf{X}, \mathbf{Y}, \beta_u, \beta_i \vert \mathbf{R}) = \sum_{u}\alpha r_{ui}\mathbf{y}_i - \frac{(1+\alpha r_{ui})\mathbf{y}_i \exp (\mathbf{x}_u\mathbf{y}^T_i + \beta_u + \beta_i))}
{1+\exp (\mathbf{x}_u\mathbf{y}^T_i + \beta_u + \beta_i))} - \lambda \mathbf{x}_u\\
\frac{\partial}{\partial \beta_i}\log p(\mathbf{X}, \mathbf{Y}, \beta_u, \beta_i \vert \mathbf{R}) = \sum_{u}\alpha r_{ui} - \frac{(1+\alpha r_{ui}) \exp (\mathbf{x}_u\mathbf{y}^T_i + \beta_u + \beta_i)))}{1+\exp (\mathbf{x}_u\mathbf{y}^T_i + \beta_u + \beta_i)))}\\
\frac{\partial}{\partial \beta_u}\log p(\mathbf{X}, \mathbf{Y}, \beta_u, \beta_i \vert \mathbf{R}) = \sum_{u}\alpha r_{ui} - \frac{(1+\alpha r_{ui}) \exp (\mathbf{x}_u\mathbf{y}^T_i + \beta_u + \beta_i)))}{1+\exp (\mathbf{x}_u\mathbf{y}^T_i + \beta_u + \beta_i)))}
$$


---
# **4. Experimental Study**
It compared Logistic MF with [IMF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4781121)(Implicit Matrix Factorization) model using MPR evaluation metric. The most improvement of Logistic MF to IMF comes from its ability to outperform IMF under a fewer number of latent factors.  

### 4.1 Dataset
Dataset consisting of user listening behavior from music streaming service Spotify (tracked the number of times they listened to each artist in $I$)

### 4.2 Evaluation Metric
chosen a recall based evaluation metric MPR(Mean Percentage Ranking) due to the lack of negative feedback. It evaluates a user's satisfaction with an ordered list of recommended items. Lower values of MPR are more desirable as they indicate that the user listened to artists higher in their predicted lists.

* $MPR = \frac{\sum_{ui}r_{ui}^{test} rank_{ui}}{\sum_{ui}r^{test}_{ui}}$
* $rank_{ui} = \text{percentile ranking of item } i \text{ for user }u$

$rank_{ui} = 0$% signifies that $i$ is predicted as the highest recommended item for $u$. Similarly, $rank_{ui} = 100$% signifies that $i$ is predicted as the lowest recommended item for $u$.

### 4.3 Result
![result](https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/LMFresult.PNG?raw=true)


### Reference
* Johnson, Christopher C. "Logistic matrix factorization for implicit feedback data." _Advances in Neural Information Processing Systems_ 27 (2014): 78.
* Hu, Yifan, Yehuda Koren, and Chris Volinsky. "Collaborative filtering for implicit feedback datasets." _2008 Eighth IEEE International Conference on Data Mining_. Ieee, 2008.
