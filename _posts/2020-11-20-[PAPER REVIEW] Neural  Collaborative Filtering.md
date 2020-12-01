---
layout: post
title: PAPER REVIEW-Neural Collaborative Filtering
tags: [paper review, recommender system, matrix factorization, Deep Learning, implicit feedback]
use_math: true
---

***He, Xiangnan, et al. "Neural collaborative filtering." Proceedings of the 26th international conference on world wide web. 2017.***

This [paper](https://dl.acm.org/doi/pdf/10.1145/3038912.3052569?casa_token=8SoRxfF0gD8AAAAA:bb8-vwZJGm7ChbPo2R-cevA5h39gnYZNXjOeX4_wcl2FOxSCicp83cv-qxzyaJDJeZGMWQLNpQlcJk0) points out that despite the effectiveness of Matrix Factorization for collaborative filtering, its performance can be hindered by the simple choice of the interaction function — inner product which simply combines the multiplication of latent features linearly.

So, it presents a general framework named NCF, short for Neural network-based Collaborative Filtering, by replacing the inner product with a neural architecture that can learn an arbitrary function from data.  

And it shows significant improvements of proposed NCF framework over the state-of-the-art methods by experiments on two real-world datasets(movielens, pinterest).



---
# **1. Learning from Implicit Data**

<p align="center">
$\text{Let }M = \text{the number of users}$, $N = \text{the number of items}$
</p>


* Define the user-item interaction matrix from users' implicit feedback as :  

$$
\mathbf{Y} = [y_{ui}] \in \mathbb{R}^{M \times N}\\
y_{ui} = \begin{cases}
1, \quad \mbox{if interaction (user u and item i) is observed}\\
0, \quad \mbox{o.w}
\end{cases}
$$

* Predictive model

$$
\hat{y}_{ui} = f(u, i \vert \Theta)
$$

 where $\Theta$ denotes model parameters and $f$ denotes the function that maps model parameter to the predicted score(termed as an interaction function in this paper).


---
# **2. Matrix Factorization**
MF can be deemed as a linear model of latent factors. See FM post [here](https://ddoeunn.github.io/2020/11/01/PAPER-REVIEW-Factorization-Machines.html).  
* Predictive model

$$
\text{Let } \mathbf{p}_u = \text{the latent vector for user }u, \mathbf{q}_i = \text{the latent vector for item }i\\
\hat{y}_{ui} = f(u, i \vert \mathbf{p}_u, \mathbf{q}_i) = \mathbf{p}^T_u \mathbf{q}_i = \underset{k=1}{\sum^K}p_{uk}q_{ik}\\
\text{where }K=\text{the dimension of the latent space}
$$

* How the inner product function can limit the expressiveness of MF?  

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/ncf_figure1.PNG?raw=true"  alt="limitation example"  width="400">
</p>

From data matrix (a), $u_4$ is most similar to $u_1$, followed by $u_3$, and lastly $u_2$. (jaccard coefficient similarity : $s_{41}(0.6) > s_{43}(0.4) >s_{42}(0.2)$ ). However, in the latent space (b), placing $\mathbf{p}_4$ closest to $\mathbf{p}_1$  makes $\mathbf{p}_4$ closer to $\mathbf{p}_2$ than $\mathbf{p}_3$, incurring a large ranking loss.


* How to resolve the issue?  
  + How about to use a large number of latent factors $K$?  
$\rightarrow$ it may adversely hurt the generalization of the model (e.g. overfitting the data)  
  + This paper address the limitation by learning the interaction function using DNNs from data.  

---
# **3. Neural Collaborative Filtering**


### **3.1 General Framework**
<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/ncf_figure2.PNG?raw=true"  alt="general framework"  width="400">
</p>  

#### Predictive Model

$$
\begin{align*}
\hat{y}_{ui} &= f(\mathbf{P}^T\mathbf{v}^U_u, \mathbf{Q}^T\mathbf{v}^I_i \vert \mathbf{P}, \mathbf{Q}, \Theta_f)\\
f(\mathbf{P}^T\mathbf{v}^U_u, \mathbf{Q}^T\mathbf{v}^I_i) &= \phi_{out}(\phi_{X}(\cdots \phi_{2}(\phi_{1}(\mathbf{P}^T\mathbf{v}^U_u, \mathbf{Q}^T\mathbf{v}^I_i))))
\end{align*}
$$

where $\mathbf{P} \in \mathbb{R}^{M \times K}, \mathbf{Q} \in \mathbb{R}^{N \times K}$ denotes the latent factor matrix for users and items, respectively  
$\Theta_f$ denotes the model parameters of the interaction function $f$.

---
#### Learning NCF
Probabilistic approach for learning the pointwise NCF that pays special attention to the binary property of implicit data. By employing a probabilistic treatment for NCF, can address recommendation with implicit feedback as a binary classification problem.  

$$
p(\mathcal{Y}, \mathcal{Y}^{-} \vert \mathbf{P}, \mathbf{Q}, \Theta_f) = \underset{(u, i) \in \mathcal{Y}}{\prod}\hat{y}_{ui}
\underset{(u, i) \in \mathcal{Y}^{-}}{\prod}\left(1- \hat{y}_{ui} \right)
$$

$$
\begin{align*}
L &= -\underset{(u, i) \in \mathcal{Y}}{\sum}\log \hat{y}_{ui} - \underset{(u, j) \in \mathcal{Y}^{-}}{\sum}\log (1-\hat{y}\_{uj})\\
&=-\underset{(u, i) \in \mathcal{Y}\cup\mathcal{Y}^{-}}{\sum}y_{ui} \log \hat{y}_{ui} + (1-y\_{ui})\log (1-\hat{y}\_{ui})
\end{align*}
$$

where $\mathcal{Y}$ = the set of observed interactions in $\mathbf{Y}$,
$\mathcal{Y^{-}}$ = the set of negative instances, all(or sampled from) unobserved interactions.


---


#### Input Layer  
* Consists of two feature vectors $\mathbf{v}^U_u$ and $\mathbf{v}^I_i$ that describe user $u$ and item $i$, respectively.
* Transforming it to a binarized sparse vector with one-hot encoding.  

#### Embedding Layer
* Fully connected layer that projects the sparse representation to a dense vector
* The obtained user(item) embedding can be seen as the latent vector for user(item) in the context of latent factor model.

#### Neural CF Layers
* The user embedding and item embedding are fed into a multi-layer neural architecture "Neural CF Layers"
* The dimension of the last hidden layer $X$ determines the model's capability.  

#### Output Layer
* The predicted score $\hat{y}_{ui}$
* Training is performed by minimizing the pointwise loss between $\hat{y}\_{ui}$
and its target value $y_{ui}$.



---
### **3.2 Generalized Matrix Factorization**
* Show how MF can be interpreted as a special case of NCF framework  

<p align="center">
Let the user latent vector $\mathbf{p}_u$ be $\mathbf{P}^T\mathbf{v}^U_u$ and item latent vector $\mathbf{q}_i$ be $\mathbf{Q}^T\mathbf{v}^I_i$
</p>

<p align="center">  
Define the mapping function of the first neural CF layer as :  
</p>

$$
\phi_1 (\mathbf{p}_u, \mathbf{q}_i) = \mathbf{p}_u \odot \mathbf{q}_i
$$

<p align="center">
then project the  vector to the output layer :    
</p>

$$
\hat{y}_{ui} = a_{out}(\mathbf{h}^T(\mathbf{p}_u \odot \mathbf{q}_i))
$$

This paper implement a generalized version of MF under NCF that uses the sigmoid function as $a_{out}$ and learns $\mathbf{h}$ from data with the log loss.


---
### **3.3 Multi-Layer Perceptron**
* MLP model under NCF framework is defined as :  

$$
\begin{align*}
\mathbf{z}_1 &= \phi_{1}(\mathbf{p}_u, \mathbf{q}_i) = \begin{bmatrix}
\mathbf{p}_u \\
\mathbf{q}_i
\end{bmatrix}\\
\phi_{2}(\mathbf{z}_1) &= a_2(\mathbf{W}^T_2 \mathbf{z}_1 + \mathbf{b}_2)\\
\vdots\\
\phi_{L}(\mathbf{z}_{L-1}) &= a_{L}(\mathbf{W}^T_{L}\mathbf{z}_{L-1} + \mathbf{b}_{L})\\
\hat{y}_{ui} &= \sigma(\mathbf{h}^T \phi_{L}(\mathbf{z}_{L-1}))
\end{align*}
$$

where $\mathbf{W}_x, \mathbf{b}_x$ and $a_x$ denote the weight matrix, bias vector, and activation function for the $x$-th layer's perceptron.  



---
### **3.4 Fusion of GMF and MLP**

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/ncf_figure3.PNG?raw=true"  alt="ncf model"  width="400">
</p>  


* GMF : applies a linear kernel to model the latent feature interactions.
* MLP : uses a non-linear kernel to learn the interaction function from data.
* Allow GMF and MLP to learn separate embeddings, and combine the 2 models by concatenating their last hidden layer.  

$$
\phi^{GMF} = \mathbf{p}^G_u \odot  \mathbf{q}^G_i\\
\phi^{MLP} = a_L(\mathbf{W}^T_L (a_{L-1}(\cdots a_{2}(\mathbf{W}^T_{2}
\begin{bmatrix}
\mathbf{p}^M_u \\
\mathbf{q}^M_i
\end{bmatrix} + \mathbf{b}_2
)\cdots)) + \mathbf{b}_L)\\
\hat{y}_{ui} = \sigma(\mathbf{h}^T
\begin{bmatrix}
\phi^{GMF}\\
\phi^{MLP}
\end{bmatrix}
  )
$$



---
# **4. Experiments**
#### Dataset  
* Movielens(1m) ;transformed each entry as 0 or 1 along whether the user has rated the item
* Pinterest ;each interaction denotes whether the user has pinned the image to own board

#### Evaluaion metrics - performance of a ranked list  
* HR(Hit Ratio) : measures whether the test item is present on the top-10 list.
* NDCG(Normalized Discounted Cumulative Gain) : accounts for the position of the hit by assigning higher scores to hits at top ranks.

#### Result

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/ncf_figure5.PNG?raw=true"  alt="result" >
</p>

---
# **Summary**
1. It present a neural network architecture to model latent features of users and items and devise a general framework NCF for collaborative filtering based on neural networks.

2. It show that MF can be interpreted as a specialization of NCF and utilize a multi-layer perceptron to endow NCF modelling with a high level of non-linearities.

3. Extensive experiments on two real-world datasets show significant improvements of NCF framework over the state-of-the-art methods.



---
## Reference
[[1]](https://dl.acm.org/doi/pdf/10.1145/3038912.3052569?casa_token=FfITqllG5HMAAAAA:rI_bL7aiSwK9r061e8X7_SEIpBIfLd8_MGB3yMrIlj53dzlfvN97S_qZDIgKPepzSjjy5cFHEUgCgvY) He, Xiangnan, et al. "Neural collaborative filtering." Proceedings of the 26th international conference on world wide web. 2017.
