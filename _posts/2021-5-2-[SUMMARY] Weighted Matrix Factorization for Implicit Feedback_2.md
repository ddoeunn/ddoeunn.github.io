---
layout: post
title: Matrix Factorization for Item Recommendation from Implicit Feedback - (2)
tags: [recommender system, implicit feedback, wmf, lmf, bpr, matrix factorization]
use_math: true
---


***Matrix Factorization Methods for Implicit Feedback***
See previous post -> [here!](https://ddoeunn.github.io/2021/05/02/SUMMARY-Weighted-Matrix-Factorization-for-Implicit-Feedback_1.html)

* Weighted Regularized Matrix Factorization
* Probabilistic Matrix Factorization
* Pairwise Matrix Factorization -> See next post [here!]()

---
# **1. Weighted Regularized Matrix Factorization**

Basic idea of Weighted Regularized Matrix Factorization (WRMF) is to assign smaller weights to the unobserved instances than the observed. The weights are related to the concept of confidence. As not interacting with an item can result from other reasons than not liking it, the zero values of $y_{ui}$  have low confidence. For example, a user might be unaware of the existence of the item, or unable to consume it due to its price or limited availability. Unobserved instances are a mixture of negative and unknown feedback.  

Also, interacting with an item can be caused by a variety of reasons that differ from liking it. For example, a  user may buy an item as gift for someone else, despite the user does not like the item. Thus it can be thought that there are also different confidence levels among the items that the user interacted with.

A common objective function of the WRMF method uses the squared error loss function and the L2 norm regularization function as follows.
Several strategies for defining weight function $w(u, i)$ have been proposed.



$$
L = \underset{u}{\sum}\underset{i}{\sum} w(u, i) \cdot (y_{ui} - \mathbf{p}_u^T \mathbf{q}_i)^2
           + \lambda_U \lVert \boldsymbol{P} \rVert^2
           + \lambda_I \lVert \boldsymbol{Q} \rVert^2
$$

* Weighting Strategies

|                 	|    $(u, i) \in S$   	|                                           $(u, i) \in (U \times I) \setminus S$                                           	|     reference     	|
|:---------------:	|:-------------------:	|:-------------------------------------------------------------------------------------------------------------------------:	|:-----------------:	|
|                 	| $1 + \alpha r_{ui}$ 	|                                                             1                                                             	|  [Hu et al. [2008]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4781121&casa_token=mOQtn7Era6QAAAAA:xiWhrafDmzfE4Xbmw9CW952M_6zG1_O8Yd464auijGfTSZWhV7RsSwNZjfkz8liOQ3Z5uvyoLrA&tag=1) 	|
|     uniform     	|          1          	|                                                    $\alpha \in (0, 1)$                                                    	| [Pan et al. [2008]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4781145&casa_token=bvg2l_uHBGsAAAAA:L7rgVq31qixKFqhy__TElXjo2FPrryQ2-96PSGxSpgLTPGDWmcMPEmBaxOl4_4QiNP_p58uig04) 	|
|  user-oriented  	|          1          	|                                          $\alpha \underset{i \in I_u}\sum y_{ui}$                                         	| [Pan et al. [2008]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4781145&casa_token=bvg2l_uHBGsAAAAA:L7rgVq31qixKFqhy__TElXjo2FPrryQ2-96PSGxSpgLTPGDWmcMPEmBaxOl4_4QiNP_p58uig04) 	|
|  item-oriented  	|          1          	|                                 $\alpha \left(m - \underset{u \in U_i}\sum y_{ui} \right)$                                	| [Pan et al. [2008]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4781145&casa_token=bvg2l_uHBGsAAAAA:L7rgVq31qixKFqhy__TElXjo2FPrryQ2-96PSGxSpgLTPGDWmcMPEmBaxOl4_4QiNP_p58uig04) 	|
| item populairty 	|          1          	| $c_0 \frac{f_i^\alpha}{\sum_{j=1}^{n}f_j^\alpha}$ where $f_i = \frac{\lvert U_i \rvert}{\sum^{n}_{j=1}\lvert U_j \rvert}$ 	|  [He et al. [2016]](https://dl.acm.org/doi/pdf/10.1145/2911451.2911489?casa_token=fI979xUDJ9oAAAAA:VEW3psI52dFmrbANoGTAkxq6sCsE22PnvGbkMf_8nm0p6yl-kvh8FLfDS2MB-LuSUmYMiJgbg7HZ8dM) 	|


## **1.1 Weights on Observed Instances**
[Hu et al. [2008]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4781121&casa_token=mOQtn7Era6QAAAAA:xiWhrafDmzfE4Xbmw9CW952M_6zG1_O8Yd464auijGfTSZWhV7RsSwNZjfkz8liOQ3Z5uvyoLrA&tag=1) first introduced WRMF method which measures confidence with the value of $r_{ui}$. It assumes that as $r_{ui}$ grows, we have a strong indication that the user indeed likes the item. It assigns $1 + \alpha r_{ui}$ to observed instances and $1$ to unobserved instances. The rate of increase is controlled by the constant $\alpha$.

$$
    w(u, i) = \begin{cases}
                1 + \alpha r_{ui}, & \text{if } (u, i) \in S \\
                1, & \text{otherwise}.
              \end{cases}
$$

---
## **1.2 Weights on Unobserved Instances**
 [Pan et al. [2008]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4781145&casa_token=bvg2l_uHBGsAAAAA:L7rgVq31qixKFqhy__TElXjo2FPrryQ2-96PSGxSpgLTPGDWmcMPEmBaxOl4_4QiNP_p58uig04) introduced three strategies to assign weights to unobserved instances -uniform/user-oriented/item-oriented.

First, uniform strategy assigns weights uniformly by the assumption that an unobserved instance has an equal chance over all users and items. It assigns 1 to observed instances and $\alpha \in (0, 1)$ to unobserved instances.

$$
    w(u, i) = \begin{cases}
                1, & \text{if } (u, i) \in S \\
                \alpha \in (0, 1), & \text{otherwise}.
              \end{cases}
$$

Second, user-oriented strategy assigns weights by the assumption that if a user has more positive examples, it is more likely that the user does not like the other items, that is, the unobserved instance for this user is negative with higher probability. It assigns 1 to observed instances and $\alpha \underset{i \in I_u}{\sum}y_{ui}$, the number of all interactions of user $u$, to unobserved instances. $\alpha$ is a hyperparameter that controls the strength of weights. Notice that the weights of unobserved instances should be lower than observed instance's weight.

$$
    w(u, i) = \begin{cases}
                1, & \text{if } (u, i) \in S \\
                \alpha \underset{i \in I_u}{\sum}y_{ui}, & \text{otherwise}.
              \end{cases}
$$

Third, item-oriented strategy assigns weights by the assumption that if an item has fewer positive examples, the unobserved instance for this item is negative with higher probability. It assigns 1 to observed instances and $\alpha (m-\underset{u \in U_i}{\sum}y_{ui})$, difference between the number of all users and the number of all users who interacted with item $i$, to unobserved instances. $\alpha$ is a hyperparameter that controls the strength of weights. Notice that the weights of unobserved instances should be lower than observed instance's weight.

$$
    w(u, i) = \begin{cases}
                1, & \text{if } (u, i) \in S \\
                \alpha (m-\underset{u \in U_i}{\sum}y_{ui}), & \text{otherwise}.
              \end{cases}
$$

[He et al. [2016]](https://dl.acm.org/doi/pdf/10.1145/2911451.2911489?casa_token=fI979xUDJ9oAAAAA:VEW3psI52dFmrbANoGTAkxq6sCsE22PnvGbkMf_8nm0p6yl-kvh8FLfDS2MB-LuSUmYMiJgbg7HZ8dM) introduced a weighting strategy based on some item property. It parameterizes the confidence based on popularity of item, given by its frequency in the implicit feedback.
It assigns 1 to observed instances and $c_0\frac{f_i^\alpha}{\sum_{j=1}^{n}f_j^\alpha}$ to unobserved instances
where $f_i = \lvert U_i \rvert / \sum^{n}_{j=1}\lvert U_j \rvert$ denotes the popularity of item $i$, hyperparameter $c_0$ determines the overall weight of unobserved instances and $\alpha$ controls the significance level of popular items over unpopular ones.

$$
    w(u, i) = \begin{cases}
                1, & \text{if } (u, i) \in S \\
                c_0 \frac{f_i^\alpha}{\sum_{j=1}^{n}f_j^\alpha}, & \text{otherwise}.
              \end{cases}
$$

If $\alpha > 1$, the difference of weights between popular items and unpopular ones is strengthened. On the contrary, if $\alpha \in (0, 1)$, the difference of weights between popular items and unpopular ones is weakened and the weight of popular items is suppressed.

---
## **1.3 Learning Algorithm**

ALS(Alternating Least Squares) optimization algorithm optimizes one parameter, while leaving the other fixed and iterates this process alternatively. If the user latent factors or the item latent factors are fixed, the objective function becomes quadratic.

Let $\boldsymbol{W} = [w_{ui}]$ denote the weight matrix where  represents the weight on user  and item  instance. The objective function can be formulated in matrix form as follow.

$$
L = \lVert \boldsymbol{W} \odot (\boldsymbol{Y} - \boldsymbol{P}\boldsymbol{Q}^T) \rVert^2 + \lambda_U \lVert \boldsymbol{P} \rVert^2 + \lambda_I \lVert \boldsymbol{Q} \rVert^2
$$

And minimizing it with respect to user latent vector  is equivalent to minimizing

$$
L_u = \lVert \boldsymbol{W}^u (\boldsymbol{y}_u - \boldsymbol{Q}\boldsymbol{p}_u) \rVert^2
                        + \lambda \lVert \boldsymbol{p}_u \rVert^2
$$

where $\boldsymbol{W}^{u} = diag(w_{u1}, \cdots, w_{un})$ and $\boldsymbol{y}_u = u^{th}$ row vector of $\boldsymbol{Y}$. The minimum is where the first-order derivative equals 0. Thus, expression of $\boldsymbol{p}_u$ that minimizes the objective function can be obtained as follow.

$$
\frac{\partial}{\partial \boldsymbol{p}_u}L_u
 = (\boldsymbol{Q}^T \boldsymbol{W}^u \boldsymbol{Q} + \lambda \boldsymbol{I})
\boldsymbol{p}_u - \boldsymbol{Q}^T
\boldsymbol{W}^u \boldsymbol{y}_u = 0
$$

$$
\Rightarrow \boldsymbol{p}_u = (\boldsymbol{Q}^T \boldsymbol{W}^u \boldsymbol{Q} + \lambda \boldsymbol{I})^{-1}\boldsymbol{Q}^T \boldsymbol{W}^u \boldsymbol{y}_u
$$

Minimizing the objective function with respect to item latent vector $\boldsymbol{q}_i$ follows same process

$$
\boldsymbol{q}_i = (\boldsymbol{P}^T \boldsymbol{W}^i \boldsymbol{P} + \lambda \boldsymbol{I})^{-1}\boldsymbol{P}^T \boldsymbol{W}^i \boldsymbol{y}_i
$$

where $\boldsymbol{W}^i = diag(w_{1i}, \cdots, w_{mi})$ and $\boldsymbol{y}_i = i^{th}$ column vector of $\boldsymbol{Y}$.

---
## **Reference**
[[1]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4781121&casa_token=mOQtn7Era6QAAAAA:xiWhrafDmzfE4Xbmw9CW952M_6zG1_O8Yd464auijGfTSZWhV7RsSwNZjfkz8liOQ3Z5uvyoLrA&tag=1) Hu, Yifan, Yehuda Koren, and Chris Volinsky. "Collaborative filtering for implicit feedback datasets." 2008 Eighth IEEE International Conference on Data Mining. Ieee, 2008.  
[[2]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4781145&casa_token=bvg2l_uHBGsAAAAA:L7rgVq31qixKFqhy__TElXjo2FPrryQ2-96PSGxSpgLTPGDWmcMPEmBaxOl4_4QiNP_p58uig04) Pan, Rong, et al. "One-class collaborative filtering." 2008 Eighth IEEE International Conference on Data Mining. IEEE, 2008.  
[[3]](https://dl.acm.org/doi/pdf/10.1145/2911451.2911489?casa_token=fI979xUDJ9oAAAAA:VEW3psI52dFmrbANoGTAkxq6sCsE22PnvGbkMf_8nm0p6yl-kvh8FLfDS2MB-LuSUmYMiJgbg7HZ8dM) He, Xiangnan, et al. "Fast matrix factorization for online recommendation with implicit feedback." Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval. 2016.
