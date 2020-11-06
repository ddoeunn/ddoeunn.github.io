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
# 2. Personalized Ranking
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
Machine learning approaches for item recommenders typically create the training data from $S$ by giving pairs $(u, i) \in S$ a positive class label and all other combinations in $(U \times I) \setminus S$ a negative one.
<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/bpr_figure_1.PNG?raw=true" alt="figure1"  width="400"/>
