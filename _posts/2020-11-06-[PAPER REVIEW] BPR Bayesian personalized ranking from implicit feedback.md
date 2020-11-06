---
layout: post
title: PAPER REVIEW_BPR Bayesian personalized ranking from implicit feedback
tags: [paper review, recommender system, matrix factorization, implicit feedback, Bayesian]
use_math: true
---


This [paper](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf) presents a generic optimization criterion BPR-OPT for personalized ranking from implicit feedback that is the maximum posterior estimator derived from a Bayesian analysis of the problem. Unlike personalized ranking(also called item recommendation) from implicit feedback like Matrix Factorization(MF) or Adaptive kNN, BPR is directly optimized for ranking. It also provide a generic learning algorithm for optimizing models with respect to BPR-OPT(based on stochastic gradient descent with bootstrap sampling). And it show how to apply this method to recommender models: MF and adaptive kNN.

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
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/bpr_figure_1.PNG?raw=true" alt="figure1"  width="400"/>

This paper uses a different approach by using item pairs as training data and optimize for correctly ranking item pairs. Assume that :
* If item $i$ has been viewed by user $u$ (i.e. $(u, i) \in S$), then the user prefers this item over all other non-observed items.
* For items that have both been seen by a user, we cannot infer any preference.
* For items that have not seen yet by a user, we cannot infer any preference.

To formalize this, create training dataset $D_S : U \times I \times I$ by $D_S := \{ (u, i, j) \vert i \in I^+_u \wedge j \in I \setminus I^+_u \}$
<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/bpr_figure_2.PNG?raw=true" alt="figure1"  width="400"/>

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

Then user-specific likelihood function $p(>_u \vert \Theta)$ can be rewitten as :
$$
\prod_{u \in U}p(>_u \lvert \Theta)
= \prod_{(u, i, j) \in U \times I \times I}p(i >_u j \vert \Theta )^{\delta((u, i, j) \in D_S)}
\left(1- p(i >_u j \vert \Theta )\right)^{\delta((u, i, j) \notin D_S)}
$$
