---
layout: post
title: 추천시스템 Neural Collaborative Filtering 알고리즘 파이썬 구현
tags: [recommender system, NCF, Collaborative Filtering, Matrix Factorization, Python]
use_math: true
---

***추천시스템 NCF 알고리즘 파이썬 구현***

* 논문 -> [He, Xiangnan, et al. "Neural collaborative filtering." Proceedings of the 26th international conference on world wide web. 2017.
](https://dl.acm.org/doi/pdf/10.1145/3038912.3052569?casa_token=FfITqllG5HMAAAAA:rI_bL7aiSwK9r061e8X7_SEIpBIfLd8_MGB3yMrIlj53dzlfvN97S_qZDIgKPepzSjjy5cFHEUgCgvY)

### **목차**
0. 문제 정의  
1.  
2.  
3.  
4.  

---
# **0. 문제 정의**
본 논문에서는 암시적 피드백 (Implicit Feedback)을 이용한 추천시스템 문제를 다룬다.

암시적 피드백은 시청기록, 구매내역처럼 사용자의 선호도를 간접적으로 파악할 수 있는 데이터이다. 이와 상반되는 개념으로 명시적 피드백 (Explicit Feedback)이 있다. 명시적 피드백은 별점이나 좋아요/싫어요 버튼처럼 사용자가 직접적으로 자신의 선호를 나타낸 데이터이다.

명시적 피드백은 사용자가 직접 선호도를 나타내야 하기 때문에 데이터를 수집하기 어려운 반면에, 암시적 피드백은 사용자의 어떤 행동을 통해 데이터를 간접적으로 수집하기 때문에 상대적으로 쉽게 데이터를 수집할 수 있다. 하지만 사용자의 선호를 간접적으로만 파악할 수 있으며, 특히 직접적이고 확실한 negative feedback이 없기 때문에 본질적으로 noisy하다는 특징이 있다.

$M$명의 사용자로부터 $N$개의 아이템에 대해 수집한 암시적 피드백을 이용하여, 사용자-아이템 상호작용 행렬 (user-item interaction matrix) $\mathbf{Y} = [y_{ui}] \in \mathbb{R}^{M \times N}$를 다음과 같이 정의한다.

$$
y_{ui} = \begin{cases}
1, \quad \text{if interaction (user $u$ and item $i$) is observed}\\
0, \quad \text{o.w}
\end{cases}
$$

사용자 $u$와 아이템 $i$의 상호작용이 관찰된 경우는 아이템 $i$에 대한 사용자 $u$의 암시적 피드백이 수집된 경우를 의미한다. 여기서 주의할 점은 $y_{ui}=1$인 경우가 사용자 $u$와 아이템 $i$ 사이에 상호작용이 있음을 나타낼뿐, 실제로 해당 아이템에 대한 사용자의 "선호"를 나타내지는 않는다는 것이다. 마찬가지로 $y_{ui} = 0$인 경우도 사용자 $u$와 아이템 $i$ 사이에 상호작용이 없음을 나타낼뿐, 해당 아이템에 대한 사용자의 "비선호"를 나타내지는 않는다.

암시적 피드백을 이용한 추천 문제는 피드백이 관찰되지 않은(즉, $y_{ui} = 0$) (사용자, 아이템) 쌍의 score를 추정하는 문제이다. 추정된 score는 사용자별 아이템 순위를 지정하는 데 사용되며, 높은 순위를 기록한 아이템을 사용자에게 추천해준다.

상호작용 $y_{ui}$의 score를 추정하기 위한 모델을 다음과 같이 표기한다.  

$$
\hat{y}_{ui} = f(u, i \vert \Theta)
$$

여기서 $\hat{y}\_{ui}$는 사용자 $u$와 아이템 $i$의 상호작용 $y_{ui}$의 예측 점수(predicted score)을 나타내며, $\Theta$는 모델 파라미터, $f$는 모델 파라미터를 예측 점수에 매핑하는 함수를 나타낸다.

모델 파라미터 $\Theta$를 추정하는 방법으로는 목적함수(objective function)을 정의하고 이를 최적화하는 기계학습(machine learning) 패러다임을 따른다. 암시적 피드백을 이용한 추천시스템에서 가장 많이 쓰이는 목적함수로 pointwise loss와 pairwise loss가 있다.


---
# **1. Matrix Factorizaiton이란?**





---
# **2. Matrix Factorization의 한계점**




---
# **3. **
