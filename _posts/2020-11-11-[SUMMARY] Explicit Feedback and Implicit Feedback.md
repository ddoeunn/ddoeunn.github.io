---
layout: post
title: SUMMARY-Explicit feedback and Implicit feedback
tags: [summary, recommender system]
---

Recommender systems rely on different types of input. The types of input can be divided into "explicit" feedback and "implicit" feedback.  


* Explicit feedback includes explicit input by users regarding their interest in products.  
(e.g. ratings)
* Implicit feedback includes implicit input which indirectly reflect opinion through observing user behavior.  
(e.g.purchase history, browsing history, search patterns, or even mouse movements)

Most convenient is the high quality explicit feedback but it is not always available. There are reluctance of users to rate products, or limitations of the system that is unable to collect explicit feedback. In contrast, once the user gives approval to collect usage data, implicit feedback requires no additional explicit feedback. (e.g. ratings)

There are several characteristics of implicit feedback that contrast with explicit feedback.

##### No negative feedback
> By observing the users behavior, we can infer which items they probably like and thus chose to consume. However, it is hard to reliably infer which items a user did not like.  
For example, a user that did not watch a certain show might have done so because she dislikes the show or just because she did not know about the show or was not available to watch it.   
This fundamental asymmetry does not exist in explicit feedback where users tell us both what they like and what they dislike. Thus explicit recommenders tend to focus on the gathered information.   
But it is impossible with implicit feedback, because concentrating only on the gathered feedback will leave us with the positive feedback, greatly misrepresenting the full user profile.

##### Inherently noisy

##### Numerical value indicates confidence

##### Evaluation requires appropriate measures
