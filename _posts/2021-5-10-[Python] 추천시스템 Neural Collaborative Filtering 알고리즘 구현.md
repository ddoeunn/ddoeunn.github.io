---
layout: post
title: 추천시스템 Neural Collaborative Filtering 논문 리뷰 & 알고리즘 파이썬 구현
tags: [recommender system, NCF, Collaborative Filtering, Matrix Factorization, Python]
use_math: true
---

***추천시스템 NCF 논문 리뷰 & 알고리즘 파이썬 구현***

* 논문 -> [He, Xiangnan, et al. "Neural collaborative filtering." Proceedings of the 26th international conference on world wide web. 2017.
](https://dl.acm.org/doi/pdf/10.1145/3038912.3052569?casa_token=FfITqllG5HMAAAAA:rI_bL7aiSwK9r061e8X7_SEIpBIfLd8_MGB3yMrIlj53dzlfvN97S_qZDIgKPepzSjjy5cFHEUgCgvY)

### **목차**
1. 문제 정의  
2. Matrix Factorization - Matrix Factorization이란? / Matrix Factorization의 한계점  
3.  
4.  

---
# **1. 문제 정의**
본 논문에서는 암시적 피드백 (Implicit Feedback)을 이용한 추천시스템 문제를 다룬다.

암시적 피드백은 시청기록, 구매내역처럼 사용자의 선호도를 간접적으로 파악할 수 있는 데이터이다. 이와 상반되는 개념으로 명시적 피드백 (Explicit Feedback)이 있다. 명시적 피드백은 별점이나 좋아요/싫어요 버튼처럼 사용자가 직접적으로 자신의 선호를 나타낸 데이터이다. 암시적 피드백은 사용자의 선호를 간접적으로만 파악할 수 있으며, 특히 직접적이고 확실한 negative feedback이 없기 때문에 본질적으로 noisy하다는 특징이 있다.

$M$명의 사용자로부터 $N$개의 아이템에 대해 수집한 암시적 피드백을 이용하여, 사용자-아이템 상호작용 행렬 (user-item interaction matrix) $\mathbf{Y} = [y_{ui}] \in \mathbb{R}^{M \times N}$를 다음과 같이 정의한다.

$$
y_{ui} = \begin{cases}
1, \quad \text{if interaction (user $u$ and item $i$) is observed}\\
0, \quad \text{o.w}
\end{cases}
$$

사용자 $u$와 아이템 $i$의 상호작용이 관찰된 경우는 아이템 $i$에 대한 사용자 $u$의 암시적 피드백이 수집된 경우를 의미한다. 여기서 주의할 점은 $y_{ui}=1$인 경우가 사용자 $u$와 아이템 $i$ 사이에 "상호작용이 있음"을 나타낼뿐, 실제로 해당 아이템에 대한 사용자의 "선호"를 나타내지는 않는다는 것이다. 마찬가지로 $y_{ui} = 0$인 경우도 사용자 $u$와 아이템 $i$ 사이에 "상호작용이 없음"을 나타낼뿐, 해당 아이템에 대한 사용자의 "비선호"를 나타내지는 않는다.

암시적 피드백을 이용한 추천 문제는 피드백이 관찰되지 않은(즉, $y_{ui} = 0$) (사용자, 아이템) 쌍의 score를 추정하는 문제이다. 추정된 score는 사용자별 아이템 순위를 지정하는 데 사용되며, 사용자와 상호작용이 관찰되지 않은 아이템 중에서 높은 순위를 기록한 아이템을 사용자에게 추천해준다.

상호작용 $y_{ui}$의 score를 추정하기 위한 모델을 다음과 같이 표기한다.  

$$
\hat{y}_{ui} = f(u, i \vert \Theta)
$$

여기서 $\hat{y}\_{ui}$는 사용자 $u$와 아이템 $i$의 상호작용 $y_{ui}$의 예측 점수(predicted score)을 나타내며, $\Theta$는 모델 파라미터, $f$는 모델 파라미터를 예측 점수에 매핑하는 함수를 나타낸다. 모델 파라미터 $\Theta$를 추정하는 방법으로는 목적함수(objective function)을 정의하고 이를 최적화하는 기계학습(machine learning) 패러다임을 따른다.


---
# **2. Matrix Factorization**

## **2.1 Matrix Factorization이란?**
Matrix Factorization(행렬분해)은 추천시스템의 Collaborative Filtering(협업 필터링) 방법론 중 model-based approach에 속하는 방법이다.  

Collaborative Filtering은 "특정 아이템에 대하여 선호가 유사한 사용자들은 다른 아이템들에 대해서도 비슷한 선호를 가질 것"이라는 아이디어를 기반으로 하며, 그 중 model-based approach는 데이터에 내제되어 있는 복잡한 패턴을 발견하기 위해 다양한 모델을 활용하는 방법이다.

Matrix Factorization은 "사용자와 아이템 사이에는 사용자의 행동에 영향을 끼치는 잠재된 특성이 있을 것"이라는 아이디어를 기반으로, 사용자-아이템 상호작용 행렬을 저차원 $K$의 사용자 잠재요인 행렬 $\mathbf{P} \in \mathbb{R}^{M \times K}$와 아이템 잠재요인 행렬 $\mathbf{Q} \in \mathbb{R}^{N \times K}$으로 분해한 뒤 두 행렬의 곱으로 예측 모델을 정의한다.

$$
\mathbf{P} =
\begin{bmatrix}
  \mathbf{p}_{1}^{T}\\
  \vdots \\
  \mathbf{p}_{m}^{T}\\
\end{bmatrix}
\mathbf{Q} =
\begin{bmatrix}
  \mathbf{q}_{1}^{T}\\
  \vdots \\
  \mathbf{q}_{n}^{T}\\
\end{bmatrix}
\mathbf{p}_u =
\begin{pmatrix}
  p_{u1}\\
  \vdots\\
  p_{uK}
\end{pmatrix}
\mathbf{q}_i =
\begin{pmatrix}
  q_{i1}\\
  \vdots\\
  q_{iK}
\end{pmatrix}
$$


$$
\hat{\mathbf{Y}} = \mathbf{P}\mathbf{Q}^T
$$

Matrix Factorization 모델은 상호작용 $y_{ui}$를 다음과 같이 사용자 잠재요인 벡터 $\mathbf{p}\_{u}$와 아이템 잠재요인 벡터 $\mathbf{q}_{i}$의 내적으로 추정한다.

$$
\hat{y}_{ui} =
\mathbf{p}_{u}^T \mathbf{q}_i =
\sum_{j=1}^{K} p_{uj}q_{ij}
$$

---
## **2.2 Matrix Factorization의 한계점**
논문의 저자는 내적 같은 linear 모델은 사용자와 아이템 사이의 복잡한 관계를 표현하는 데 한계가 있다고 지적한다.  

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/ncf_figure1.PNG?raw=true"  alt="limitation example"  width="400">
</p>

위의 그림 (a)와 같은 사용자-아이템 상호작용 행렬이 있을 때, jaccard coefficient를 통해 계산한 사용자 $i$와 $j$의 유사도를 $s_{ij}$로 표기하자.

$$
s_{ij} = \frac{\lvert \mathcal{R}_i \rvert \cap \lvert \mathcal{R}_j \rvert}{\lvert \mathcal{R}_i \rvert \cup \lvert \mathcal{R}_j \rvert}
$$

$$
\text{where } \mathcal{R}_u \text{ denotes set of items that user $u$ has interacted with}
$$

사용자 1, 2, 3 사이의 유사도를 다음과 같이 구할 수 있다.

$$
s_{12} = \frac{2}{4} = 0.5, \quad s_{13} = \frac{2}{5} = 0.4, \quad s_{23} = \frac{2}{3} = 0.66
$$

$$
s_{23}(0.66) > s_{12}(0.5) > s_{13}(0.4)
$$


사용자 2와3이 가장 비슷하고, 사용자 1과3이 가장 덜 비슷하다. 위의 그림 (b)는 이런 관계를 기하학적으로 나타낸 그림이다.

linear 모델의 한계는 여기서 새로운 사용자 4가 등장했을 때 발생한다. 새로운 사용자 4와 기존 사용자 1, 2, 3과의 유사도를 계산해보면 다음과 같은 관계가 성립한다.

$$
s_{41} = \frac{3}{5} = 0.6 \quad s_{42} = \frac{1}{5} = 0.2 \quad s_{43} = \frac{2}{5} = 0.4
$$

$$
s_{41}(0.6) > s_{43}(0.4) > s_{42}(0.2)
$$

사용자 4는 사용자 1과 가장 비슷하고, 사용자 2와 가장 덜 비슷하다. 하지만 사용자 1, 2, 3이 만든 사용자 잠재 공간에 새로운 사용자 4를 나타낼 때, 4와 1을 가장 가깝게 하는 동시에 4와 2를 가장 멀게 하는 벡터 $\mathbf{p}_4$를 표현할 수 없다. 즉, 사용자와 아이템간의 복잡한 관계를 저차원의 단순한 공간에 표현하는데 한계가 있다는 것이다.


위의 예는 저차원 잠재 공간에서 복잡한 사용자-아이템 상호작용을 추정하기 위해 단순하고 고정된 내적을 사용함으로써 발생할 수 있는 Matrix Factorization의 한계를 보여준다. 저자는 이러한 한계점을 해결하기 위해 DNN을 이용한 방법을 새롭게 제시한다.



---
# **3. Neural Collaborative Filtering**


### **3.1 General Framework**

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/ncf_figure2.PNG?raw=true"  alt="general framework"  width="400">
</p>  

위의 그림은 저자가 제시한 Neural Collaborative Filtering의 general framework이다.


####  Input Layer  
* 사용자와 아이템의 one-hot encoding 벡터 $\mathbf{v}_u^U$, $\mathbf{v}_i^I$를 input으로 사용한다.  
* 예) 사용자 4와 아이템 3에 대한 input 벡터  
$\mathbf{v}_4^U = (0, 0, 0, 1, 0, ..., 0)^{\prime}$, $\mathbf{v}_3^I = (0, 0,  1, 0, ..., 0)^{\prime}$

#### Embedding Layer
* embedding은 고차원 벡터의 변환을 통해 생성할 수 있는 상대적인 저차원 공간을 가리킨다.
* embedding layer는 sparse한 input 벡터를 dense 벡터로 매핑하는 Fully Connected layer이다.
* embedding layer를 통해 가중치 행렬 $\mathbf{P}$와 $\mathbf{Q}$를 얻을 수 있다.
* $\mathbf{P}$의 $u$번째 행은 사용자 $u$를 표현하는 저차원의 dense 벡터가 되며, 이를 사용자 잠재 벡터 (user latent vector)로 사용한다.
* $\mathbf{Q}$의 $i$번째 행은 아이템 $i$를 표현하는 저차원의 dense 벡터가 되며, 이를 아이템 잠재 벡터 (item latent vector)로 사용한다.



#### Neural CF Layers
* 사용자 잠재요인 벡터 $\mathbf{P}^T \mathbf{v}_u^U$와 아이템 잠재요인 벡터 $\mathbf{Q}^T \mathbf{v}_i^I$를 concatenating한 벡터를 input으로 받아 deep neural network를 통과한다.
*  deep neural network  

$$
\phi_{X}(\cdots \phi_{2}(\phi_{1}(\mathbf{P}^T\mathbf{v}^U_u, \mathbf{Q}^T\mathbf{v}^I_i)), \cdots)
$$

#### Output Layers
* 사용자 $u$와 아이템 $i$의 상호작용 $y_{ui}$의 예측 점수(predicted score) $\hat{y}_{ui}$를 구한다.

$$
\hat{y}\_{ui} = f(\mathbf{P}^T\mathbf{v}^U_u, \mathbf{Q}^T\mathbf{v}^I_i) = \phi_{out}(\phi_{X}(\cdots \phi_{2}(\phi_{1}(\mathbf{P}^T\mathbf{v}^U_u, \mathbf{Q}^T\mathbf{v}^I_i))))
$$


---
#### Learning NCF
먼저 모델을 학습하기 위한 objective function을 정의한다. $y_{ui}$가 0 또는 1 값을 갖는 binary 데이터이기 때문에 $y_{ui}$에 베르누이 분포를 가정할 수 있다. 여기서 확률  $p(y_{ui})$는 output 레이어에서 $\phi_{out}$에 logisitc 함수를 acitivation 함수로 사용하여 값을 0과 1사이로 만든 $\hat{y}_{ui}$를 사용한다.

$$
y_{ui} \sim Bernoulli(p(y_{ui}))
$$

$$
p(y_{ui}) = \hat{y}_{ui}
$$

$\mathcal{Y}$가 상호작용이 관측된 (사용자, 아이템)집합 (즉, $\mathcal{Y} = \{(u, i) \vert y_{ui} = 1\}$)이고, $\mathcal{Y}^{-}$가 상호작용이 관측되지 않은 (사용자, 아이템)집합 (즉, $\mathcal{Y}^{-} = \{(u, i) \vert y_{ui} = 0\}$)일 때  likelihood 함수는 다음과 같다.

$$
p(\mathcal{Y}, \mathcal{Y}^{-} \vert \mathbf{P}, \mathbf{Q}, \Theta_f) =
\prod_{(u, i) \in \mathcal{Y}} {\hat{y}_{ui}}^{y_{ui}}
\prod_{(u, i) \in \mathcal{Y}^{-}}(1 - \hat{y}_{ui})^{1-y_{ui}}
$$


likelihook 함수에 negative log 변환을 한 함수를 objective function으로 정의하고, 이 함수를 최소화하는 모델 파라미터를 찾는다. 학습 알고리즘으로는 Stochastic Gradient Descent (SGD) 알고리즘을 사용한다.

$$
\begin{align*}
L &= -\underset{(u, i) \in \mathcal{Y}}{\sum}\log \hat{y}_{ui} - \underset{(u, j) \in \mathcal{Y}^{-}}{\sum}\log (1-\hat{y}_{uj})\\
&=-\underset{(u, i) \in \mathcal{Y}\cup\mathcal{Y}^{-}}{\sum}y_{ui} \log \hat{y}_{ui} + (1-y_{ui})\log (1-\hat{y}_{ui})
\end{align*}
$$




---
# **4. **
