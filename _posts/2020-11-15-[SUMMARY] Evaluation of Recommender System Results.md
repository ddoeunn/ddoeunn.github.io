---
layout: post
title: SUMMARY-Evaluation of Recommender System Results
tags: [summary, recommender system, evaluation]
use_math: true
---

I summarized evaluation measures of Recommender System. The contents about implicit feedback evaluation will be uploaded separately.

---
## **1. Quality of the predictions**  
* In order to measure the accuracy of the results of an RS, it is usual to use the calculation of some of the most common prediction error metrics
* Notations

$$
\begin{align*}
U &= \text{the set of users}\\
I &= \text{the set of items}\\
r_{ui} &= \text{rating of user }u \text{ on item }i\\
\bullet &= \text{the lack of rating}\\
r_{ui} &= \bullet \text{ ; means user }u \text{ has not rated item }i\\
p_{ui} &= \text{prediction of item }i \text{ on user }u
\end{align*}
$$

<p align="center">
Let $O_u = \{i \in I \vert p_{ui} \ne \bullet \wedge r_{ui} \ne \bullet \}$ :set of items rated by user $u$ having prediction values
</p>

---
#### 1.1 MAE : Mean Absolute Error

$$
MAE = \frac{1}{\lvert U \rvert} \underset{u \in U}{\sum}\left( \frac{1}{\lvert O_u \rvert} \underset{i \in Q_u}{\sum}\lvert p_{ui} - r_{ui} \rvert  \right)
$$

#### 1.2 RMSE : Root Mean Squared Error

$$
RMSE = \frac{1}{\lvert U \rvert} \underset{u \in U}{\sum}\sqrt{ \frac{1}{\lvert O_u \rvert} \underset{i \in Q_u}{\sum}\left( p_{ui} - r_{ui} \right)^2 }
$$

#### 1.3 Coverage  
calculates the percentage of situations in which at least one $k$-neighbor of each active user can rate an item that has not been rated yet bu that active user.

<p align="center">
Define $K_{ui} = \text{the set of neighbors of } u \text{ which have rated the item }i$
</p>
<p align="center">  
Let $C_u = \{i \in I \vert r_{ui}=\bullet \wedge K_{ui} \ne \bullet \}$ and $D_u = \{i\in I \vert r_{ui} = \bullet \}$
</p>

$$
coverage = \frac{1}{\lvert U \rvert}\underset{u \in U}{\sum}\left(100 \times \frac{\lvert C_u \rvert}{\lvert D_u \rvert}\right)
$$



---
## **2. Quality of the set of recommendations**
* The confidence of users for a certain recommender system does not depend directly on the accuracy for the set of possible predictions.
* A user gains confidence on the recommender system when this user agrees with a reduced set of recommendations made by recommender system.
* Evaluation measures obtained by making $n$ test recommendations to user $u$, taking a $\theta$ relevancy threshold : Precision / Recall / F1  
* Notations  

$$
\begin{align*}
X_u &= \text{the set of recommendations to user }u\\
Z_u &= \text{the set of }n\text{ recommendations to user }u
\end{align*}
$$

---
#### 2.1 Precision
indicates the proportion of relevant recommended items from the total number of recommended items.  

$$
precision = \frac{1}{\lvert U \rvert}\underset{u \in U}{\sum}\frac{\lvert \{i \in Z_u \vert r_{ui} \ge \theta \}  \rvert}{n}
$$

#### 2.2 Recall
indicates the proportion of relevant recommended items from the number of relevant items.

$$
recall = \frac{1}{\lvert U \rvert}\underset{u \in U}{\sum}
\frac{\lvert \{ i \in Z_u \vert r_{ui} \ge \theta \} \rvert}
{\lvert \{i \in Z_u \vert r_{ui} \ge \theta \}  \rvert +
\lvert \{i \in Z^c_u \vert r_{ui} \ge \theta \} \rvert}
$$

#### 2.3 F1
harmonic mean of precision and recall  

$$
F1 = \frac{2\times precision \times recall}{precision+recall}
$$

---
## **3. Quality of the list of recommendations: rank measure**
* When the number $n$ of recommended items is not small, users give greater importance to the first items on the list of recommendations.
* Notations  

$$
\begin{align*}
p_1, &\cdots, p_n : \text{recommendation list}\\
k &= \text{rank of the evaluated item}\\
d &= \text{default rating}\\
\alpha &= \text{the number of item on the list}
\end{align*}
$$

---
#### 3.1 HL : Half Life
assume an exponential decrease in the interest of users as they move away from the recommendations at the top.  

$$
HL = \frac{1}{\lvert U \rvert}\underset{u \in U}{\sum}\underset{i=1}{\sum^{N}}\frac{\max(r_{u, p_i}-d, 0)}
{2^{(i-1)/(\alpha -1)}}
$$

#### 3.2 DCG : Discounted Cumulative Gain
decay is logarithmic  

$$
DCG^k = \frac{1}{\lvert U \rvert}\underset{u \in U}{\sum}
\left(\underset{i=1}{\sum^k} \frac{r_{u, p_i}}{\log_2(i+1)} \right)
$$


---
## **4. Novelity and Diversity**
#### 4.1 Novelity
indicates the degree of difference between the items recommended to and known by user.

$$
novelity_i = \frac{1}{\lvert Z_u \rvert -1}
\underset{j \in Z_u}{\sum}\left( 1-sim(i, j) \right), i\in Z_u
$$


#### 4.2 Diversity
indicates the degree of differentiation among recommended items

$$
diversity_{Z_u} = \frac{1}{\lvert Z_u \rvert (\lvert Z_u \rvert -1)}\underset{i \in Z_u}{\sum}\underset{j \in Z_u, j\ne i}{\sum}(1-sim(i, j))
$$

---
### Reference  
[[1]](https://www.sciencedirect.com/science/article/pii/S0950705113001044?casa_token=Cg_uVDgFTUUAAAAA:zEeUd5PSfP6Z1szGdBS25JU7plzQ-Nha9_ADnejuJGSaruzX44PhcrZa-ss6CxFFJWnO6Xv_Em4) Bobadilla, Jesús, et al. "Recommender systems survey." Knowledge-based systems 46 (2013): 109-132.
