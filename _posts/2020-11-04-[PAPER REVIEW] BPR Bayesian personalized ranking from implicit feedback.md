---
layout: post
title: PAPER REVIEW_BPR Bayesian personalized ranking from implicit feedback
tags: [paper review, recommender system, matrix factorization, implicit feedback, Bayesian]
use_math: true
---

Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." arXiv preprint arXiv:1205.2618 (2012).

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
