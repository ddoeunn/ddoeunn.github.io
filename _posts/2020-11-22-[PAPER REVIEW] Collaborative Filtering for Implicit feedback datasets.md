---
layout: post
title: PAPER REVIEW-Collaborative Filtering for Implicit Feedback
tags: [paper review, recommender system, matrix factorization, implicit feedback]
use_math: true
---

***Hu, Yifan, Yehuda Koren, and Chris Volinsky. "Collaborative filtering for implicit feedback datasets." 2008 Eighth IEEE International Conference on Data Mining. Ieee, 2008.***

This [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4781121&casa_token=OJF5_t4aaMQAAAAA:MgeH_JXt6wG6LfOCvhJj37q1slIPpQLO46Lrs1bU_4FOsChcChOLka8JXY3eWbUOE_4GZRDFLq8&tag=1) identify unique properties of implicit feedback dataset. It propose treating the data as indication of positive and negative preference associated with vastly varying confidence levels.  This leads to a factor model which is especially tailored for implicit feedback recommenders.   

I summarized the characteristics of the implicit data and the model that reflects it (especially the confidence levels). I also recommend to see the paper ["Logistic matrix factorization for implicit feedback data"](http://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf) [(review)](https://ddoeunn.github.io/2020/11/04/PAPER-REVIEW-Logistic-Matrix-Factorization-for-Implicit-Feedback-Data.html) which compares the methods presented by this paper.



---
# **1. Characteristics of implicit feedback**
Recommender systems rely on different types of input(Explicit and Implicit). Most convenient is the high quality explicit feedback, which includes explicit input by users regarding their interest in products. However, explicit feedback is not always available. Thus, recommenders can infer user preferences from the more abundant implicit feedback, which indirectly reflect opinion through observing user behavior . Types of implicit feedback include purchase history, browsing history, search patterns, or even mouse movements.  
Characteristic of implicit feedback follow below. (See more detailed explanation [here](https://ddoeunn.github.io/2020/11/11/SUMMARY-Explicit-Feedback-and-Implicit-Feedback.html))

* no negative feedback
* inherently noisy
* numerical value indicates confidence(not preference)
* evaluation requires appropriate measures



---
# **2. Preliminaries**
#### Notations
* indexing letters for users $u, v$ and for items $i, j$
* rating :users and items are associated through $r_{ui}$ values (observations)

#### Rating - explicit vs implicit
1. Explicit ratings are typically unknown for the vast majority of user-item pairs.  
$\rightarrow$ Applicable algorithms work with the relatively few known ratings while ignoring the missing one.  

2. With implicit feedback, it would be natural to assign values to all $r_{ui}$.  
$\rightarrow$ If no action was observed, $r_{ui}$ is set to zero.



---
# **3. Model for implicit feedback**

* Formalize the notion of confidence which the $r_{ui}$ variables measure

$$
p_{ui} = \begin{cases}
1 \quad r_{ui} > 0 \\
0 \quad r_{ui} = 0
\end{cases}
$$

$$
c_{ui} = 1+\alpha r_{ui}\\
\text{or}\\
c_{ui} = 1 + \alpha \log(1 + \frac{r_{ui}}{ \epsilon })
$$


* Predictive model

$$
\hat{p_{ui}} = \mathbf{x}^T_i \mathbf{y}_i = \underset{k=1}{\sum^{f}} x_{uk}y_{ik}
$$


* Optimization problem  
Goal : to find a vector $\mathbf{x}_u \in \mathbb{R}^f$ for each user $u$, $\mathbf{y}_i \in \mathbb{R}^f$ for each item $i$

$$
\underset{\mathbf{x}^{*}, \mathbf{y}^{*}}{\min}\underset{u, i}{\sum}c_{ui}
\left( p_{ui} - \mathbf{x}_u^T \mathbf{y}_i \right)^2 +
\lambda \left( \underset{u}{\sum}\lVert \mathbf{x}_u \rVert^2 + \underset{i}{\sum}\lVert \mathbf{y}_i \rVert^2 \right)
$$

---
#### Why need varying confidence levels?  
In general, as $r_{ui}$ grows, we have a stronger indication that the user indeed likes the item.  

1. Not taking any positive action on an item can stem from many other reasons beyond not liking it.(zero values of $p_{ui}$ are associated with low confidence)
$\rightarrow$ For example, the user might be unaware of the existence of the item, or unable to consume it due to its price or limited availability.  

2. Consuming an item can also be the result of factors different from preferring it.  
$\rightarrow$ For example, a user may watch a TV show just because she is staying on the channel of the previously watched show. Or a consumer may buy an item as gift for someone else, despite not liking the item for himself.  
$\therefore$ have different confidence levels also among items that are indicated to be preferred by the user.


---
## Reference
[[1]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4781121&casa_token=RbyPeJcQA-oAAAAA:221aLXd94s255FCYJ5A2fw-Sg4LqodrWmL5GB-wiqRNCZ4D0B4F_pLrfQj1D_-osfA5WhWW4qDg&tag=1) Hu, Yifan, Yehuda Koren, and Chris Volinsky. "Collaborative filtering for implicit feedback datasets." 2008 Eighth IEEE International Conference on Data Mining. Ieee, 2008.  
[[2]](http://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf) Johnson, Christopher C. "Logistic matrix factorization for implicit feedback data." Advances in Neural Information Processing Systems 27 (2014): 78.
