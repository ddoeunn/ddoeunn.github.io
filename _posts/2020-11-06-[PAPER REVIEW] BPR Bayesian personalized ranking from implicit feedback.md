---
layout: post
title: PAPER REVIEW_BPR Bayesian personalized ranking from implicit feedback
tags: [paper review, recommender system, matrix factorization, implicit feedback, Bayesian]
use_math: true
---


This [paper](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf) presents a generic optimization criterion BPR-OPT for personalized ranking from implicit feedback that is the maximum posterior estimator derived from a Bayesian analysis of the problem. Unlike personalized ranking(also called item recommendation) from implicit feedback like [Matrix Factorization(MF)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4781121&casa_token=mfl1oU80aesAAAAA:pwVQVY_a2BEmxXAJGf0Y7UxU8IsVKvk2nhCE4Fm07oUD8FAB5k9aPMlC9EynZ83VD1ScaFuupA4&tag=1) or Adaptive kNN, BPR is directly optimized for ranking. It also provide a generic learning algorithm for optimizing models with respect to BPR-OPT(based on stochastic gradient descent with bootstrap sampling). And it show how to apply this method to recommender models: MF and adaptive kNN.

---
# **1. Implicit Feedback**
In real world scenarios most user feedback is not explicit but implicit. Implicit feedback is tracked automatically, like monitoring clicks, view times, purchases, etc. Thus it is much easier to collect, because the user has not to express his taste explicitly. Interesting about implicit feedback systems is that only positive observations are available. The non-observed user-item pairs are a mixture of real negative feedback and missing values.
* real negative feedback : the user is not interested in buying the item
* missing values : the user might want to buy the item in the future.

---
# **2. Personalized Ranking**
The task of personalized ranking is to provide a user with a ranked list of items. An example is an online shop that wants to recommend a personalized ranked list of items that the user might want to buy.

### 2.1 Formalization

$$
U = \text{the set of all users}\\
I = \text{the set of all items}\\
S \subseteq U \times I\\
I_u^+ := \{ i \in I : (u, i) \in S \}\\
U_i^+ := \{ u \in I : (u, i) \in S \}\\
$$

The task of the recommender system is to provide the user with a personalized total ranking $>_u \subset I^2$ of all items, where $>_u$ has to meet the properties of a total order:

* $\forall i, j \in I : i \ne j \Rightarrow i >_u j \vee j >_u i $ (totality)
* $\forall i, j \in I : i >_u j \vee j >_u i \Rightarrow i=j$ (antisymmetry)
* $\forall i, j, k \in I : i>_u j \vee j >_u k \Rightarrow i >_u k$ (transitivity)

### 2.2 Analysis of the problem setting
Machine learning approaches for item recommenders typically create the training data from $S$ by giving pairs $(u, i) \in S$ a positive class label and all other combinations in $(U \times I) \setminus S$ a negative one. That means a model with enough expressiveness (that can fit the training data exactly) cannot rank at all as it predicts only 0s.

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/bpr_figure_1.PNG?raw=true" alt="figure1"  width="400">
</p>

This paper uses a different approach by using item pairs as training data and optimize for correctly ranking item pairs. Assume that :
* If item $i$ has been viewed by user $u$ (i.e. $(u, i) \in S$), then the user prefers this item over all other non-observed items.
* For items that have both been seen by a user, we cannot infer any preference.
* For items that have not seen yet by a user, we cannot infer any preference.

To formalize this, create training dataset $D_S : U \times I \times I$ by  

$$ D_S := \{ (u, i, j) \vert i \in I^+_u \wedge j \in I \setminus I^+_u \}$$

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/bpr_figure_2.PNG?raw=true" alt="figure1"  width="400">
</p>
This approach has 2 advantages:
* It also gives information to unobserved items so that they can be learned indirectly.
* It can also rank unobserved items.

---
# **3. BPR: Bayesian Personalized Ranking**
### 3.1 BPR Optimization criterion
The Bayesian formulation ofnding the correct personalized ranking for all items $i \in I$ is to maximize the following posterior probability where $\Theta$ represents the parameter vector of an arbitrary model class.

$$
p(\Theta \vert >_u) \propto p(>_u \vert \Theta)p(\Theta)
$$

Assume that :
* all users act independently of each other.
* the ordering of each pair of items $(i, j)$ for a specific user is independent of the ordering of every other pair.

---
#### Likelihood

$$ >_u := (u, i, j) \in D_S\\
\delta((u, i, j) \in D_S) \sim Bernoulli(p(i >_u j))\\
\text{where } \delta \text{ is the indicator function.}
$$

* Due to the totality and antisymmetry, simplified to :

$$
\prod_{u \in U}p(>_u \vert \Theta)
= \prod_{(u, i, j) \in D_S}p(i >_u j \vert \Theta)
$$

* Then user-specific likelihood function $p(>_u \vert \Theta)$ can be rewitten as :

$$
\prod_{u \in U}p(>_u \lvert \Theta)
= \prod_{(u, i, j) \in D_S}p(i >_u j \vert \Theta )^{\delta((u, i, j) \in D_S)}
\left(1- p(i >_u j \vert \Theta )\right)^{\delta((u, i, j) \notin D_S)}
$$

* Define the individual probability that a user prefers item $i$ to item $j$ as :

$$
p(i >_u  j \vert \Theta)
:= \sigma(\hat{x}_{uij}(\Theta))\\
\text{where } \sigma(x) := \frac{1}{1+e^{-x}} \text{: logistic sigmoid}
$$

$\hat{x}_{uij}(\Theta)$ is an real-valued function of the model parameter vector $\Theta$ which captures the relationship between user $u$, item $i$ and item $j$.




---
#### Prior
Prior density of $\Theta$:

$$
\begin{align*}
\Theta &\sim N(\mathbf{0}, \lambda_\Theta I) \\
p(\Theta) &= \frac{1}{(2 \pi )^{\frac{d}{2}}\lvert \lambda_\Theta I \rvert^{\frac{1}{2}}}\exp \left( -\frac{1}{2}\left(\Theta - \mathbf{0} \right)^T \lambda_\Theta I^{-1}\left(\Theta - \mathbf{0} \right)  \right)\\
 &\propto \exp \left(-\frac{1}{2} \Theta^T \left(\frac{1}{\lambda_{\Theta}}I \right)\Theta \right)\\
&= \exp \left(-\frac{1}{2 \lambda_{\Theta}} \Theta^T \Theta \right)\\
&=\exp \left(-\frac{1}{2 \lambda_{\Theta}} \lVert \Theta \rVert^2 \right)
\end{align*}
$$



---
Formulate the maximum posterior estimator to derive generic optimization criterion for personalized ranking BPR-OPT.


$$
\begin{align*}
\text{BPR-OPT} :&= \ln p(\Theta \vert >_u)\\
&= \ln p(>_u \vert \Theta)p(\Theta)\\
&= \ln \prod_{(u, i, j) \in D_S}\sigma(\hat{x}_{uij}(\Theta))p(\Theta)\\
&= \underset{(u, i, j) \in D_S}\sum \ln \sigma(\hat{x}_{uij}(\Theta)) + \ln p(\Theta)\\
&= \underset{(u, i, j) \in D_S}\sum \ln \sigma(\hat{x}_{uij}(\Theta))-
 \lambda_{\Theta} \lVert \Theta \rVert^2\\
\end{align*}$$

 where $\lambda_{\Theta}$ are model specific regularization parameters.


---
### 3.2 BPR Learning Algorithm
* Gradient of BPR-OPT with respect to the model parameter :  

$$
\begin{align*}
\frac{\partial \text{BPR-OPT}}{\partial \Theta}
&=\underset{(u, i, j) \in D_S}{\sum}\frac{\partial }{\partial \Theta} \ln \sigma(\hat{x}_{uij}(\Theta))-
 \lambda_{\Theta} \frac{\partial }{\partial \Theta}  \lVert \Theta \rVert^2\\
 &= \underset{(u, i, j) \in D_S}{\sum}\frac{-e^{-\hat{x}_{uij}}}
 {1 + e^{-\hat{x}_{uij}}} \cdot \frac{\partial }{\partial \Theta}
 \hat{x}_{uij}(\Theta)
 - \lambda_{\Theta} \Theta
\end{align*}$$

* Update rule of full gradient descent with learning rate $\alpha$

$$
\Theta \leftarrow \Theta - \alpha \frac{\partial \text{BPR-OPT}}{\partial \Theta}
$$

* Update rule of stochastic gradient descent with learning rate $\alpha$

$$
\Theta \leftarrow \Theta + \alpha \left(\frac{e^{-\hat{x}_{uij}}}
{1 + e^{-\hat{x}_{uij}}} \cdot \frac{\partial }{\partial \Theta}
\hat{x}_{uij}(\Theta) + \lambda_{\Theta} \Theta  \right)
$$

* Optimization algorithm with bootstrapping based SGD with learning rate $\alpha$ and regularization $\lambda_{\Theta}$

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/bpr_algorithm.PNG?raw=true" alt="algorithm"  width="400">
</p>

* Why SGD?
> A typical approach that traverses the data item-wise or user-wise will lead to poor convergence as there are so many consecutive updates on the same user-item pair. To solve this issue we suggest to use a stochastic gradient descent algorithm that chooses the triples randomly.

* Why bootstrapping?
> With SGD(uniformly random sampling) the chances to pick the same user-item combination in consecutive update steps is small. We suggest to use a bootstrap sampling approach with replacement because stopping can be performed at any step.

---
# **4. Learning MF model with BPR**  
Because in our optimization we have triples $(u, i, j) \in D_S$ decompose the estimator $\hat{x}_{uij}$ and define it as:  

$$
\hat{x}_{uij} := \hat{x}_{ui} - \hat{x}_{uj}
$$

#### Matrix Factorization  
Target matrix $X$ is approximated by the matrix product of 2 low-rank matrices $W_{\lvert U \rvert \times k}$ and $H_{\lvert I \rvert \times k}$ where $k$ is the dimensionality of the approximation.  

$$
\hat{X} := WH^T\\
\hat{x}_{ui} = \left< w_u, h_i \right>
= \sum_{f=1}^{k} w_{uf}\cdot h_{if}
$$

#### Model Parameter  

$$
\Theta = (W, H)
$$

#### Gradient of $\hat{x}_{uij}$

$$
\begin{align*}
\frac{\partial}{\partial w_{uf}}\hat{x}_{uij} &= (h_{if} - h_{jf})\\
\frac{\partial}{\partial h_{if}}\hat{x}_{uij} &= w_{uf}\\
\frac{\partial}{\partial h_{jf}}\hat{x}_{uij} &= -w_{uf}
\end{align*}
$$

Also, use three regularization constants: $\lambda_W$ for user featrues $W$, $\lambda_{H^+}$ for positive updates on $h_{if}$ and $\lambda_{H^-}$ for negative updates on $h_{jf}$.

---
# **5. Evaluation**
It compared BPR-MF with SVD-MF and WR-MF using AUC evaluation metric. The model dimensions are increased from 8 to 128 dimensions.
#### Datasets
Rossmann dataset and Netflix dataset(remove the rating scores and predict if a user is likely to rate a movie)

#### Evaluation metric

$$
\text{AUC} = \frac{1}{\lvert U \rvert}\underset{u}{\sum}\frac{1}{\lvert E(u) \rvert}
\underset{(i, j) \in E(u)}{\sum} \delta (\hat{x}_{ui} > \hat{x}_{uj})  
$$

where the evaluation pairs per user $u$ are:

$$
E(u):=\{(i, j) \vert (u, i) \in S_{test} \wedge (u, j) \notin (S_{test} \cup S_{train})  \}
$$

(split train set $S_{train}$ and $S_{test}$ by removing one entry from $I_u^+$ per user $u$)

#### Result

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/bpr_result.PNG?raw=true" alt="result"  width="400">
</p>

---
### Reference
* Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." arXiv preprint arXiv:1205.2618 (2012).
* Hu, Yifan, Yehuda Koren, and Chris Volinsky. "Collaborative filtering for implicit feedback datasets." 2008 Eighth IEEE International Conference on Data Mining. Ieee, 2008.
* Mnih, Andriy, and Russ R. Salakhutdinov. "Probabilistic matrix factorization." Advances in neural information processing systems. 2008.
