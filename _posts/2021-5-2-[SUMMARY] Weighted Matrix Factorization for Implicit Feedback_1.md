---
layout: post
title: Matrix Factorization for Item Recommendation from Implicit Feedback - (1)
tags: [recommender system, implicit feedback, wmf, lmf, bpr, matrix factorization]
use_math: true
---


***Matrix Factorization Methods for Item Recommendation from Implicit Feedback***

One of the key methods of personalized recommendation is Collaborative Filtering (CF) which uses users' preferences for item based on their past interaction with items. CF approach has two categories, memory-based and model-based. Among model-based approaches, matrix factorization (MF) which projects users and items into shared latent factor space of reduced dimensionality is the most popular.  

[Sarwar et al., 2000](https://apps.dtic.mil/sti/pdfs/ADA439541.pdf) first applied SVD, well known matrix factorization technique, for CF as an alternative approach of memory-based method which has weakness for large, sparse rating data. MF model had become popular by showing good performance(RMSE) in rating prediction of Netflix Prize.  

Recommender system task can be divided into two categories. One is the aforementioned rating prediction, and the other is item recommendation. In the rating prediction task, which uses users' ratings belonging to explicit feedback (e.g., a user gave 5 stars to a movie), CF algorithms attempt to predict user ratings for items they have not yet rated. The item recommendation, which usually uses implicit feedback (e.g., a user watches a video), is the task to generating a ranking list for each user over as yet not interacted items.   

Most early CF algorithms had been proposed for rating prediction tasks. But in recent years, the focus of recommender system research has shifted from explicit feedback problems to implicit feedback problems ([Bayer et al., 2017](https://dl.acm.org/doi/pdf/10.1145/3038912.3052694?casa_token=zpea3-79L_AAAAAA:SL5EghSNkGA9k6pAJQhcbigCyopz70Qua20_t4zP9DrBBM9JbC7-CqqOnF6HKH18ICXa0beQkP6O2bU)). It is because most of the signal that a user provides about user's preferences is implicit. And implicit feedback data is much cheaper to obtain than explicit feedback, because it comes with no extra cost for the user and thus is available on a much larger scale ([Bayer et al., 2017](https://dl.acm.org/doi/pdf/10.1145/3038912.3052694?casa_token=zpea3-79L_AAAAAA:SL5EghSNkGA9k6pAJQhcbigCyopz70Qua20_t4zP9DrBBM9JbC7-CqqOnF6HKH18ICXa0beQkP6O2bU)).  

According to this shift, variety of MF methods for item recommendation with implicit feedback have been proposed. So, I summarize these variety of MF methods for item recommendation with implicit feedback, explore the overall process of item recommendation and apply to real data analysis.


---
# **1. Problem Setup and Notations**

$$
U = \{u_1, \cdots, u_m\}  \text{ ; set of } m \text{ users} \\
I = \{i_1, \cdots, i_n\}  \text{ ; set of } n \text{ items} \\
\boldsymbol{R} = [r_{ui}] \in \boldsymbol{R}^{m \times n} \text{ ; user-item implicit feedback matrix} \\
S = \{(u, i) \vert r_{ui}\text{ is observed.}  \forall u \in U, i \in I\} \\
$$


Our goal is to find a subset of interesting items from a set of items $I$ for a user $u \in U$ by ranking the items.  

Let $\boldsymbol{R} = [r_{ui}] \in \boldsymbol{R}^{m \times n}$ be a user-item implicit feedback matrix where $r_{ui}$ represents the observations of user $u$'s actions for item $i$. If no action was observed, the element is set to blank.  

Let $S \subseteq U \times I$ denote the set of all user-item pairs whose implicit feedback was observed. We define the user-item interaction matrix $\boldsymbol{Y}=[y_{ui}]$ treating the observed implicit feedback as positive instances and the unobserved as negatives, where the value of $y_{ui}$ is assigned as follow.  

$$
y_{ui} =
\begin{cases}
 1, & \mbox{if }(u, i) \in S \\
 0, & \mbox{if }(u, i) \in (U \times I) \setminus S
\end{cases}
$$

---
# **2. Matrix Factorization**

 Matrix factorization is one of the dimensionality reduction methods which provides a dense low-dimensional representation in terms of latent factors. Matrix factorization models map both users and items to a joint latent factor space of dimensionality , such that user-item interactions are modeled as inner products in that space ([Koren et al., 2009](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5197422&casa_token=HSj7PaZKdOsAAAAA:n1apwxxhhfMjau17bUZscoKw0tzGInSwhqoefSC_dfuJ4XIEo1DmOH21aqKzZQw9NKYBU9-5MJU&tag=1)).  

Let $\boldsymbol{P} \in \boldsymbol{R}^{m \times k}$ and $\boldsymbol{Q} \in \boldsymbol{R}^{n \times k}$ denote latent factor matrix of users and items, respectively.

$$
\boldsymbol{P} =
\begin{bmatrix}
  \boldsymbol{p}_{1}^{T}\\
  \vdots \\
  \boldsymbol{p}_{m}^{T}\\
\end{bmatrix}
\boldsymbol{Q} =
\begin{bmatrix}
  \boldsymbol{q}_{1}^{T}\\
  \vdots \\
  \boldsymbol{q}_{n}^{T}\\
\end{bmatrix}
\boldsymbol{p}_u =
\begin{pmatrix}
  p_{u1}\\
  \vdots\\
  p_{uk}
\end{pmatrix}
\boldsymbol{q}_i =
\begin{pmatrix}
  q_{i1}\\
  \vdots\\
  q_{ik}
\end{pmatrix}
$$


$\boldsymbol{p}_u \in \boldsymbol{R}^{k}$, $u$-th row vector of $\boldsymbol{P}$, denotes the latent factor vector for user $u$. Similarly $\boldsymbol{q}_i \in \boldsymbol{R}^{k}$, $i$-th row of $\boldsymbol{Q}$, denotes the latent factor vector for item $i$.  

For a given user $u$, the elements of $\boldsymbol{p}_u$ measures the extent of interest the user has in items that are high on the corresponding factors and for a given item $i$, the elements of $\boldsymbol{q}_i$ measures the extent to which the item possesses those factor ([Koren et al., 2009](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5197422&casa_token=HSj7PaZKdOsAAAAA:n1apwxxhhfMjau17bUZscoKw0tzGInSwhqoefSC_dfuJ4XIEo1DmOH21aqKzZQw9NKYBU9-5MJU&tag=1)).  

The score of interaction between user $u$  and item $i$, which is denoted by  $\hat{y}_{ui}$, is estimated by inner product of user and item latent factor vectors which captures the interaction between user and item.

$$
\hat{\boldsymbol{Y}} = \boldsymbol{P}\boldsymbol{Q}^T \\
\hat{y}_{ui} =
\boldsymbol{p}_{u}^T \boldsymbol{q}_i =
\sum_{j=1}^{k} p_{uj}q_{ij}
$$  

Some users may tend to interact with different types of items, while others only interact with smaller subsets. Similarly, some items are popular and expectations for interacting with a wide user base are high, while others are less popular and expectations for interacting with a small user base are high. This tendency can be modeled as follow

$$
\hat{y}_{ui} =
\boldsymbol{p}_{u}^T \boldsymbol{q}_i + \beta_u + \beta_i
$$

where $\beta_u$ and $\beta_i$ represent bias term of user $u$ and item $i$ respectively. Note that it is also possible to add only one of user bias and item bias term.

---
# **3. Learning Objectives for Implicit Feedback**
The problem of estimating the scores can be solved as supervised machine learning problem by formulating the objective function to learn the parameters and optimizing it.  

Item recommendation task with implicit feedback requires to formulate the objective function reflecting the characteristics of implicit feedback, especially the absence of negative feedback. The most important difference is that the objective function for explicit feedback uses only the set of known instances for learning but the objective function for implicit feedback uses both observed and unobserved instances to reflect the absence of negative feedback.

There are two types of objective functions most commonly used in item recommendation with implicit feedback. - pointwise loss and pairwise loss.

---
## **3.1 Pointwise Loss**
Similar to rating prediction using explicit feedback, estimating the exact value of the score can be considered important. As a extension of work on rating prediction, pointwise loss follows the framework of minimizing the loss between $\hat{y}\_{u i}$ and $y\_{u i}$.

A general form of pointwise loss can be defined as follow.  

$$
L(\Theta)
    = \underset{u \in U}{\sum} \underset{i \in I}{\sum} w(u, i) l(\hat{y}_{ui}(\Theta), y_{ui}) + \lambda (\Theta)
$$

where $\Theta$ is model parameters, $w$ is a weight function, $l$ is a loss function and $\lambda$ is a regularization function.

---
## **3.2 Pairwise Loss**
For pairwise loss, it is not primary interest to estimate the exact value of the score. Instead the relative order of scores between the observed item and the unobserved item for each user is important. The idea is that the observed item should have a higher score than the unobserved item. Thus the pairwise loss compares the scores of all pairs of observed and unobserved items for a user  and seeks to maximize the difference of the scores between observed and unobserved items.  

A general form of pairwise loss can be defined as follow  

$$
L(\Theta)
    =   \underset{(u, i) \in S}{\sum} \underset{j \in I \setminus I_u}{\sum}
    w(u, i, j) l(\hat{y}_{uij}(\Theta), 1) + \lambda (\Theta)
$$

where $\Theta$ is model parameters, $w$ is a weight function, $l$ is a loss function and $\lambda$ is a regularization function.
Here, $\hat{y}_{uij}(\Theta)$ is a real-valued function which captures the relative order between user $u$, item $i$ and item $j$.


**See next post "Matrix Factorization Methods for Implicit Feedback" ->**[ here!](https://ddoeunn.github.io/2021/05/02/SUMMARY-Weighted-Matrix-Factorization-for-Implicit-Feedback_2.html)


---
## **Reference**
[[1]](https://apps.dtic.mil/sti/pdfs/ADA439541.pdf) Sarwar, Badrul, et al. Application of dimensionality reduction in recommender system-a case study. Minnesota Univ Minneapolis Dept of Computer Science, 2000.  
[[2]](https://dl.acm.org/doi/pdf/10.1145/3038912.3052694?casa_token=zpea3-79L_AAAAAA:SL5EghSNkGA9k6pAJQhcbigCyopz70Qua20_t4zP9DrBBM9JbC7-CqqOnF6HKH18ICXa0beQkP6O2bU) Bayer, Immanuel, et al. "A generic coordinate descent framework for learning from implicit feedback." Proceedings of the 26th International Conference on World Wide Web. 2017.  
[[3]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5197422&casa_token=HSj7PaZKdOsAAAAA:n1apwxxhhfMjau17bUZscoKw0tzGInSwhqoefSC_dfuJ4XIEo1DmOH21aqKzZQw9NKYBU9-5MJU&tag=1) Koren, Yehuda, Robert Bell, and Chris Volinsky. "Matrix factorization techniques for recommender systems." Computer 42.8 (2009): 30-37.
