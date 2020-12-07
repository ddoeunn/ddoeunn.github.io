---
layout: post
title: PAPER REVIEW-Fast Matrix Factorization with Implicit Feedback
tags: [paper review, recommender system, matrix factorization, Deep Learning, implicit feedback]
use_math: true
---

***He, Xiangnan, et al. "Fast matrix factorization for online recommendation with implicit feedback." Proceedings of the 39th International ACM SIGIR conference on Research and Development in Information Retrieval. 2016.***  

This [paper](https://dl.acm.org/doi/pdf/10.1145/2911451.2911489?casa_token=LzXRdzA852oAAAAA:e-kgDaoWTuz3RvQCyrEoxezcc24UidtpQ6MhyJQlvNa2e9V9vWxuhOxF4lLfFLKGH6lDuhLXKSLGxiQ) address two issues in learning Matrix Factorization models from implicit feedback. First, assigning a uniform weight to the missing data to reduce computational complexity due to the large space of unobserved feedback. Second, fail to keep up with the dynamic nature of online data.  

To address these issues, this paper propose a new learning algorithm based on the element-wise Alternating Least Square(eALS) for efficiently optimizing a MF model with variably-weighted missing data.


 
