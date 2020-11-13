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

---
There are several characteristics of implicit feedback that contrast with explicit feedback.

#### 1. No negative feedback
> * By observing the users behavior, we can infer which items they probably like and thus chose to consume. However, it is hard to reliably infer which items a user did not like.  
>* For example, a user that did not watch a certain show might have done so because she dislikes the show or just because she did not know about the show or was not available to watch it.   
>* This fundamental asymmetry does not exist in explicit feedback where users tell us both what they like and what they dislike. Thus explicit recommenders tend to focus on the gathered information.   
>* But it is impossible with implicit feedback, because concentrating only on the gathered feedback will leave us with the positive feedback, greatly misrepresenting the full user profile.

#### 2. Inherently noisy
>* While we passively track the users behavior, we can only guess their preferences and true motives.
>* For example, we may view purchase behavior for an individual, but this does not necessarily indicate a positive view of the product. The item may have been purchased as a gift, or perhaps the user was disappointed with the product.



#### 3. Numerical value indicates confidence
>* Whereas the numerical value of explicit feedback indicates preference.  
>* Systems based on explicit feedback let the user express their level of preference, e.g. a star rating between 1 (“totally dislike”) and 5 (“really like”).
>* On the other hand, numerical values of implicit feedback describe the frequency of actions, e.g., how much time the user watched a certain show, how frequently a user is buying a certain item  
>* A larger value is not indicating a higher preference in implicit feedback.
>* However, a recurring event is more likely to reflect the user opinion than one time event that might be caused by various reasons that have nothing to do with user preferences.

#### 4. Evaluation requires appropriate measures
>* In the traditional setting where a user is specifying a numeric score, there are clear metrics such as mean squared error to measure success in prediction.  
>* However with implicit models we have to take into account availability of the item, competition for the item with other items, and repeat feedback.


---
### Reference  
[[1]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4781121&casa_token=q-jqrctSVAoAAAAA:kj52K48yJaSnAJGpb7J1aJOX9Q0UQLcL5J9JQYloqxtZE_TISYsOR2XE95IIyBxpdfAZlppIK70&tag=1) Hu, Yifan, Yehuda Koren, and Chris Volinsky. "Collaborative filtering for implicit feedback datasets." 2008 Eighth IEEE International Conference on Data Mining. Ieee, 2008.
