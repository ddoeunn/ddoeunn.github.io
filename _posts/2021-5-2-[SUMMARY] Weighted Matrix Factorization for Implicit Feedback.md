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






---
## **Reference**
[[1]](https://apps.dtic.mil/sti/pdfs/ADA439541.pdf) Sarwar, Badrul, et al. Application of dimensionality reduction in recommender system-a case study. Minnesota Univ Minneapolis Dept of Computer Science, 2000.  
[[2]](https://dl.acm.org/doi/pdf/10.1145/3038912.3052694?casa_token=zpea3-79L_AAAAAA:SL5EghSNkGA9k6pAJQhcbigCyopz70Qua20_t4zP9DrBBM9JbC7-CqqOnF6HKH18ICXa0beQkP6O2bU) Bayer, Immanuel, et al. "A generic coordinate descent framework for learning from implicit feedback." Proceedings of the 26th International Conference on World Wide Web. 2017.  
