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
* Pairwise Matrix Factorization

---
# **1. Weighted Regularized Matrix Factorization**

Basic idea of Weighted Regularized Matrix Factorization (WRMF) is to assign smaller weights to the unobserved instances than the observed. The weights are related to the concept of confidence. As not interacting with an item can result from other reasons than not liking it, the zero values of $y_{ui}$  have low confidence. For example, a user might be unaware of the existence of the item, or unable to consume it due to its price or limited availability. Unobserved instances are a mixture of negative and unknown feedback.  

Also, interacting with an item can be caused by a variety of reasons that differ from liking it. For example, a  user may buy an item as gift for someone else, despite the user does not like the item. Thus it can be thought that there are also different confidence levels among the items that the user interacted with.

A common objective function of the WRMF method uses the squared error loss function and the L2 norm regularization function as follows.
Several strategies for defining weight function $w(u, i)$ have been proposed.



$$
L = \underset{u}{\sum}\underset{i}{\sum} w(u, i) \cdot (y_{ui} - \boldsymbol{p}_u^T \boldsymbol{q}_i)^2
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

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/mf_algorithm1.PNG?raw=true" width=350>
</p>

---
# **2. Probabilistic Matrix Factorization**
## **2.1 Logistic Matrix Factorization**

[Johnson, 2014](http://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf) proposed probabilistic matrix factorization for implicit feedback which models the probability that a user chooses an item by a logistic function.

It assume a Bernoulli distribution on $y_{ui}$, which indicates whether the user $u$ interacts with the item $i$ and defines the probability that user $u$ interacts with item $i$ as follow

$$
\hat{y}_{ui} =
\boldsymbol{p}_{u}^T \boldsymbol{q}_i + \beta_u + \beta_i
$$


$$
p(y_{ui} \vert \boldsymbol{p}_{u}, \boldsymbol{q}_i, \beta_u, \beta_i)
= \sigma(\hat{y}_{ui})
$$

$$
\text{where } \sigma(x) = 1/(1 + e^{-x}) \text{ logistic sigmoid function. }
$$


It also applied concept of confidence as in WRMF by assigning $\alpha r_{ui}$ to observed instances and 1 to unobserved. Increasing $\alpha$ places more weight on the observed instances while decreasing $\alpha$ places more weight on the unobserved. With assumption that all instances of $\boldsymbol{Y}$ are independent, the likelihood of $\boldsymbol{Y}$ given model parameters is obtained as follow.

$$
L(\boldsymbol{Y} \vert \boldsymbol{P}, \boldsymbol{Q}, \boldsymbol{\beta}) = \prod_{u, i} p(y_{ui} \vert \boldsymbol{p}\_u, \boldsymbol{q}\_i, \beta_i, \beta_j)^{\alpha r_{ui}}
    (1 -  p(y_{ui} \vert \boldsymbol{p}_u, \boldsymbol{q}_i, \beta_i, \beta_j))
$$

And place a zero mean Gaussian priors on user and item latent factors.

$$
p(\boldsymbol{P} \vert \sigma^2_U) = \prod_{u} \mathcal{N}(\boldsymbol{p}_u \vert \sigma^2_U \boldsymbol{I})
$$

$$
  p(\boldsymbol{Q} \vert \sigma^2_I) = \prod_{i} \mathcal{N}(\boldsymbol{q}_i \vert \sigma^2_I \boldsymbol{I})
$$

The log of the posterior distribution, which is objective function to be maximized, is obtained as follow by replacing constant terms with a scaling parameter $\lambda$.

$$
L = \underset{u}\sum \underset{i}\sum \alpha r_{ui} \hat{y}_{ui}
        - (1 + \alpha r_{ui}) \ln (1 + \exp(\hat{y}_{ui}))
        - \frac{\lambda}{2} \left( \lVert \boldsymbol{p}_u  \rVert^2 + \lVert \boldsymbol{q}_i  \rVert^2 \right)
$$

It performed alternating gradient ascent algorithm to find a local maximum of the objective function, which optimizes one parameter while leaving the other fixed and iterates this process alternatively. The partial derivatives are obtained as follow.

$$
\frac{\partial L}{\partial \boldsymbol{p}_u}
            = \underset{i}\sum \alpha r_{ui}\boldsymbol{q}_i
                - \frac{\boldsymbol{q}_i (1 + \alpha r_{ui}) \exp (\boldsymbol{p}_u^T \boldsymbol{q}_i + \beta_u + \beta_i)}
                        {1 + \exp (\boldsymbol{p}_u^T \boldsymbol{q}_i + \beta_u + \beta_i)}
                - \lambda \boldsymbol{p}_u
$$

$$
\frac{\partial L}{\partial \boldsymbol{q}_i}
            = \underset{u}\sum \alpha r_{ui} \boldsymbol{p}_u
                - \frac{\boldsymbol{p}_u (1 + \alpha r_{ui}) \exp (\boldsymbol{p}_u^T \boldsymbol{q}_i + \beta_u + \beta_i)}
                        {1 + \exp (\boldsymbol{p}_u^T \boldsymbol{q}_i + \beta_u + \beta_i)}
                - \lambda \boldsymbol{q}_i
$$

$$
\frac{\partial L}{\partial \beta_u}
            = \underset{i}\sum \alpha r_{ui}
                - \frac{(1 + \alpha r_{ui}) \exp (\boldsymbol{p}_u^T \boldsymbol{q}_i + \beta_u + \beta_i)}
                        {1 + \exp (\boldsymbol{p}_u^T \boldsymbol{q}_i + \beta_u + \beta_i)}
$$

$$
\frac{\partial L}{\partial \beta_i}
= \underset{u} \sum \alpha r_{ui} - \frac{(1 + \alpha r_{ui})\exp (\boldsymbol{p}_u^T \boldsymbol{q}_i + \beta_u + \beta_i)}{1 + \exp (\boldsymbol{p}_u^T \boldsymbol{q}_i + \beta_u + \beta_i)}
$$

---
# **3. Pairwise Matrix Factorization**
Pairwise matrix factorization methods directly optimize its model parameters for ranking based on pairs of items instead of scoring single items. It is derived from the idea that item recommendation is a personalized ranking task, which gives one individual ranking per user.  

 A common objective function of pairwise matrix factorization which is to be maximized is formulated as follow

$$
L(\Theta) = \underset{(u, i) \in S}\sum \underset{j \in I \setminus I_u}\sum
                     w(u, i, j) \cdot \ln \sigma(\hat{y}_{uij}(\Theta)) - \lambda(\Theta)
$$

where $\hat{y}_{uij}(\Theta)$ is a real-valued function which captures the relative order between user $u$, item $i$ and item $j$.

[Rendle et al, 2012](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf) first argued that because item recommendation is a personalized ranking task, the optimization should be tailored for ranking and proposed Bayesian Personalized Raking (BPR). Since BPR was proposed, various methods based on BPR have been proposed.

|      	|                   $\hat{y}_{ui}$                   	|          $\hat{y}_{uij}$          	|  $w(u, i, j)$ 	|       Reference       	|
|:----:	|:--------------------------------------------------:	|:---------------------------------:	|:-------------:	|:---------------------:	|
|  BPR 	|      $\boldsymbol{p}^T_{u}\boldsymbol{q}_{i}$      	|   $\hat{y}_{ui} - \hat{y}_{uj}$   	|       1       	|  [Rendle et al. [2012]](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf) 	|
| WBPR 	| $\boldsymbol{p}^T_{u}\boldsymbol{q}_{i} + \beta_i$ 	|   $\hat{y}_{ui} - \hat{y}_{uj}$   	| $w_u w_i w_j$ 	| [Gantner et al. [2012]](http://proceedings.mlr.press/v18/gantner12a/gantner12a.pdf) 	|
| GBPR 	| $\boldsymbol{p}^T_{u}\boldsymbol{q}_{i} + \beta_i$ 	| $\hat{y}_{G_{ui}} - \hat{y}_{uj}$ 	|       1       	|  [Pan and Chen [2013]](http://www.comp.hkbu.edu.hk/~lichen/download/IJCAI2013_Pan.pdf)  	|


## **3.1 Bayesian Personalized Ranking**
[Rendle et al, 2012](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf) formalized the task of recommender system as providing the user $u$ with a personalized total ranking $>_u \subset I^2$. $>_u$ represents the desired personalized ranking for all items for user $u$ and meets the properties of a total order.  

* (totality) $\forall i, j \in I : i \ne j \Rightarrow i >_u j \vee j >_u i$
* (antisymmetry) $\forall i, j \in I : i >_u j \wedge j >_u i \Rightarrow i=j$
* (transitivity) $\forall i, j, k \in I : i >_u j \wedge j >_u k \Rightarrow i >_u k$

 It introduced BPR-OPT, a generic optimization criterion for personalized ranking which is derived by a Bayesian analysis, and applied it to matrix factorization model optimization.   

 There are two assumptions to derive BPR-OPT. The first assumption is individual pairwise preference over two items. It assumes that a user $u$ prefers an item $i$ to an item $j$, if the user-item pair $(u, i)$ is observed and $(u, j)$ is not observed (i.e., $i >_u j$). The second assumption is independence between two users. It assumes that the ordering of each pair of items $(i, j)$ for a specific user is independent of the ordering of every other pair.  

The Bayesian formulation of the task to find the personalized ranking is to maximize the posterior probability

$$
p(\Theta \vert >_u) \propto p(>_u \vert \Theta)p(\Theta)
$$

where $\Theta$ is the model parameters.

By the second assumption, user-specific likelihood function $ p(>_u \vert \Theta)$ can be rewritten as a product of single densities and by the totality and antisymmetry properties, it can be simplified as follow

$$
\underset{u \in U} \prod p(>_u \vert \Theta) &= \underset{(u, i, j) \in U \times I \times I}\prod p(i >_u j \vert \Theta)^{\delta((u, i, j)\in D_s)} \cdot (1 - p(i >_u j \vert \Theta))^{\delta((u, i, j) \notin D_s)}\\
  &= \underset{(u, i, j) \in D_s}\prod p(i >_u j \vert \Theta)
$$

where $D_s = \{(u, i, j) \vert i \in I_u \wedge j \in I \setminus I_u \}$ and $\delta$ is the indicator function.  

 Define the individual probability that a user  prefers item  to item as

 $$
p(i >_u j) = \sigma(\hat{y}_{ui}(\Theta)-\hat{y}_{uj}(\Theta))
 $$

 $$
\text{where } \sigma(x) = 1/(1 + e^{-x})
 $$


Assume the prior density as $\Theta \sim N(\boldsymbol{0}, \lambda_\Theta \boldsymbol{I})$ , a normal distribution with zero mean and variance-covariance matrix $\lambda_\Theta \boldsymbol{I}$. Then the maximum posterior to derive BPR-OPT is formulated as follow.

$$
\begin{align*}
     \text{BPR-OPT} &= \ln p(\Theta \vert >_u)\\
       &= \ln p(>_u \vert \Theta)p(\Theta) \\
       &= \ln \underset{(u, i, j) \in D_s} \prod \sigma(\hat{y}_{ui}(\Theta)-\hat{y}_{uj}(\Theta)) p(\Theta) \\
       &= \underset{(u, i, j) \in D_s} \sum \ln \sigma(\hat{y}_{ui}(\Theta)-\hat{y}_{uj}(\Theta)) - \lambda_\Theta \lVert \Theta \rVert^2
\end{align*}
$$

 The objective function to be minimized for learning matrix factorization model based on BPR-OPT can be formulated as follow ,where $\hat{y}_{ui} = \boldsymbol{p}^T_{u} \boldsymbol{q}_i$

$$
L = \underset{(u, i) \in S}{\sum}\underset{j \in I \setminus I_u}{\sum} -\ln \sigma (\hat{y}_{ui}-\hat{y}_{uj}) + \lambda_{U} \lVert \boldsymbol{p}_u \rVert^2 + \lambda_{I} \lVert \boldsymbol{q}_i \rVert^2 + \lambda_{J} \lVert \boldsymbol{q}_j \rVert^2
$$


 Due to the large number of pairs in $D_s$ ,[Rendle et al, 2012](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf) optimizes it with stochastic gradient descent (SGD) algorithm. Let $l_{uij}$ be the loss for one sample $(u, i, j) \in D_s$. Then update rule for parameters can be formulated as follow.

 $$
\boldsymbol{p}_u \leftarrow
\boldsymbol{p}_u -\alpha\{ - (1 - \sigma(\hat{y}_{ui} - \hat{y}_{uj}))
(\boldsymbol{q}_i - \boldsymbol{q}_j) + \lambda_U \boldsymbol{p}_u \}
 $$

 $$
\boldsymbol{q}_i \leftarrow
\boldsymbol{q}_i - \alpha \{ - (1 - \sigma(\hat{y}_{ui} - \hat{y}_{uj})) \boldsymbol{p}_u + \lambda_I \boldsymbol{q}_i \}
 $$

 $$
\boldsymbol{q}_i \leftarrow
\boldsymbol{q}_i - \alpha \{(1 - \sigma(\hat{y}_{ui} - \hat{y}_{uj})) \boldsymbol{p}_u + \lambda_J \boldsymbol{q}_j \}
 $$

 <p align="center">
 <img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/mf_algorithm2.PNG?raw=true" width=350>
 </p>

---
## **3.2 Weighted BPR**

[Gantner et al. [2012]](http://proceedings.mlr.press/v18/gantner12a/gantner12a.pdf)  introduced Weighted BPR (WBPR), which add weight term on BPR. Weight function is defined as follow

$$
w(u, i, j) = w_u w_i w_j
$$

$$
\begin{cases}
  w_u &= \frac{1}{\lvert I_u \rvert}  \\
  w_i &= 1 \\
  w_j &= \sum_{u\in U} y_{uj}
\end{cases}
$$

It assign 1 to observed instances and $\sum_{u\in U} y_{uj}$ to unobserved instances, which is based on global popularity of the item. And $w_u$ is a weight that balances the contribution of each user.  

Following BPR-OPT ([Rendle et al, 2012](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf)) process, the objective function of WBPR is obtained as follow where $\boldsymbol{p}^T_{u}\boldsymbol{q}_{i} + \beta_i$.

$$
L = \underset{(u, i) \in S}{\sum}\underset{j \in I \setminus I_u}{\sum} -w_u w_i w_j \ln \sigma (\hat{y}_{ui}-\hat{y}_{uj}) + \lambda_{U} \lVert \boldsymbol{P} \rVert^2 + \lambda_{I} \lVert \boldsymbol{Q} \rVert^2 + \lambda_{B} \lVert \boldsymbol{\beta} \rVert^2
$$

---
## **3.3 Group BPR**  

[Pan and Chen, 2013](http://www.comp.hkbu.edu.hk/~lichen/download/IJCAI2013_Pan.pdf) pointed out that the two assumptions made in BPR may not always hold in real application. First, a user $u$ may potentially prefer an item $i$ to an item $j$, though the user $u$ has expressed positive feedback on item $i$ instead of item $j$. Second, two users may be correlated, and their joint likelihood can not be decomposed into two independent likelihood. So it defined two types of preferences (individual preference and group preference) and proposed a new assumption.  

Individual preference, which is denoted as $\hat{y}_{ui}$ above, is a preference score of a user on an item. Group preference is an overall preference score of a group of users on an item.  

Let $G_i \subseteq U_i$ denote a group of users who share the same positive feedback to item $i$. Then a group preference of users from group $G_i$ is defined as average of individual preferences in the group.

$$
\hat{y}_{G_i} = \frac{1}{\lvert G_i \rvert}
\underset{w \in G_i}{\sum}\hat{y}_{wi}
$$

And then, it assumes that the group preference of  $G_i \subseteq U_i$ on an item $i$ is more likely to be stronger than the individual preference of user in the group on item $j$, if the user-item pair $(u, i)$ is observed and $(u, j)$ is not observed (i.e., $\hat{y}_{G_i} > \hat{y}_{uj}$).  

To unify the effect of the two types of preferences, group preference and individual preference, combine them linearly as follow

$$
\hat{y}_{G_{ui}} = \rho \hat{y}_{G_i} + (1 - \rho) \hat{y}_{ui}, \quad 0 \le \rho \le 1
$$

where $\hat{y}_{G_i}$ denotes the fused preference of group preference and individual preference and $\rho$ is a tradeoff parameter used to fuse the two preferences.  

Following BPR-OPT ([Rendle et al, 2012](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf)) process, the objective function of GBPR is obtained as follow where $\boldsymbol{p}^T_{u}\boldsymbol{q}_{i} + \beta_i$.

$$
L = \underset{(u, i) \in S}{\sum}\underset{j \in I \setminus I_u}{\sum} -\ln \sigma (\hat{y}_{G_{ui}}-\hat{y}_{uj}) + \lambda_{U} \underset{w \in G_i}{\sum}\lVert \boldsymbol{p}_w \rVert^2 + \lambda_{I} (\lVert \boldsymbol{q}_i \rVert^2 +  \lVert \boldsymbol{q}_j \rVert^2)
$$

The main difference between GBPR and BPR is that GBPR introduced richer interactions among users in the group who share the same positive feedback to an item and linearly combined it with individual preference.

---
## **Reference**
[[1]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4781121&casa_token=mOQtn7Era6QAAAAA:xiWhrafDmzfE4Xbmw9CW952M_6zG1_O8Yd464auijGfTSZWhV7RsSwNZjfkz8liOQ3Z5uvyoLrA&tag=1) Hu, Yifan, Yehuda Koren, and Chris Volinsky. "Collaborative filtering for implicit feedback datasets." 2008 Eighth IEEE International Conference on Data Mining. Ieee, 2008.  
[[2]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4781145&casa_token=bvg2l_uHBGsAAAAA:L7rgVq31qixKFqhy__TElXjo2FPrryQ2-96PSGxSpgLTPGDWmcMPEmBaxOl4_4QiNP_p58uig04) Pan, Rong, et al. "One-class collaborative filtering." 2008 Eighth IEEE International Conference on Data Mining. IEEE, 2008.  
[[3]](https://dl.acm.org/doi/pdf/10.1145/2911451.2911489?casa_token=fI979xUDJ9oAAAAA:VEW3psI52dFmrbANoGTAkxq6sCsE22PnvGbkMf_8nm0p6yl-kvh8FLfDS2MB-LuSUmYMiJgbg7HZ8dM) He, Xiangnan, et al. "Fast matrix factorization for online recommendation with implicit feedback." Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval. 2016.  
[[4]](http://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf) Johnson, Christopher C. "Logistic matrix factorization for implicit feedback data." Advances in Neural Information Processing Systems 27.78 (2014): 1-9.  
[[5]](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf) Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." arXiv preprint arXiv:1205.2618 (2012).  
[[6]](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf) Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." arXiv preprint arXiv:1205.2618 (2012).  
[[7]](http://proceedings.mlr.press/v18/gantner12a/gantner12a.pdf) Gantner, Zeno, et al. "Personalized ranking for non-uniformly sampled items." Proceedings of KDD Cup 2011. PMLR, 2012.  
[[8]](http://www.comp.hkbu.edu.hk/~lichen/download/IJCAI2013_Pan.pdf) Pan, Weike, and Li Chen. "Gbpr: Group preference based bayesian personalized ranking for one-class collaborative filtering." Twenty-Third International Joint Conference on Artificial Intelligence. 2013.  
