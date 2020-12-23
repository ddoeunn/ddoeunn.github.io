---
layout: post
title: PAPER REVIEW-VBPR:Visual Bayesian Personalized Ranking from Implicit Feedback
tags: [paper review, recommender system, deep learning, matrix factorization, implicit feedback]
use_math: true
---

***He, Ruining, and Julian McAuley. "VBPR: visual bayesian personalized ranking from implicit feedback." arXiv preprint arXiv:1510.01784 (2015).***


Recommender systems typically rely on 2 types of inputs which can be divided into "explicit" and "implicit" feedback. Implicit feedback includes implicit input which indirectly reflect opinion through observing user behavior(e.g.purchase history, browsing history, search patterns, or even mouse movements) and explicit feedback includes explicit input by users regarding their interest in products.(e.g. ratings)  

In order to model user feedback in large, realworld datasets, Matrix Factorization (MF) approaches have been proposed to uncover the most relevant latent dimensions in both explicit and implicit feedback settings.  

---
This [paper](https://arxiv.org/pdf/1510.01784.pdf) introduce a model that incorporate **visual features** of the items for the task of personalized ranking on **implicit feedback** datasets using Matrix Factorization approach.     


The goal here is to ask :
* whether it is possible to uncover the visual dimensions that are relevant to people's opinions, and if so,
* whether such 'visual preferecne' models shall lead to improved performance at tasks like personalized ranking.

To start with the conclusion, for the above two questions, it says that by learning the visual dimensions people consider when selecting products
* we will be able to alleviate cold start issues (which caused by sparsity of real datasets in Matrix Factorization models).
* help explain recommendations in terms of visual signals, produce personalized rankings that are more consistent with users' preferences.


---
# **1. Problem Formulation**  
Our objective is to generate for each user $u$ a personalized ranking of those items about which they haven't yet provided feedback ($\mathcal{I} \backslash \mathcal{I}^{+}_u$) by predicting the score($\hat{x}_{u, i}$) user $u$ gives to item $i$

- Notations

| Notation                   	| Explanation                                              	|
|----------------------------	|----------------------------------------------------------	|
| $\mathcal{U}, \mathcal{I}$ 	| user set, item set                                       	|
| $\mathcal{I}^{+}_u$        	| positive item set of user $u$                            	|
| $\hat{x}_{u, i}$           	| predicted score user $u$ gives to item $i$               	|
| $K$                        	| dimension of latent factors                              	|
| $D$                        	| dimension of visual factors                              	|
| $F$                        	| dimension of Deep CNN features                           	|
| $\alpha$                   	| global offset (scalar)                                   	|
| $\beta_u, \beta_i$         	| user $u$'s bias, item $i$'s bias (scalar)                	|
| $\gamma_u, \gamma_i$       	| latent factors of user $u$, item $i$ ($K \times 1$)      	|
| $\theta_u, \theta_i$       	| visual factors of user $u$, item $i$ ($F \times 1$)      	|
| $f_i$                      	| DeepCNN visual features of item $i$ ($F \times 1$)       	|
| $\mathbf{E}$               	| $D \times F$ embedding matrix                            	|
| $\beta^{\prime}$           	| visual bias vector (visual bias = $\beta^{\prime T}f_i$) 	|



---
# **2. Preference Predictor**
### **2.1 Basic Matrix Factorization Model**

$$
\hat{x}_{u, i} = \alpha + \beta_u + \beta_i + \gamma^T_u \gamma_i
$$

The inner product $\gamma^T_u \gamma_i$ encodes the "compatibility" between the user $u$ and the item $i$, i.e., the extent to which the user's latent "preferences" are aligned with the products' "properties".  
Although theoretically latent factors are able to uncover any relevant dimensions, one major problem it suffers from is the existence of "cold" items in the system, about which there are too few associated observations to estimate their latent dimensions.   

So this paper propose to partition rating dimensions into visual factors and latent(non-visual) factors.


---
### **2.2 Partition Rating Dimensions into Visual Factors and Non-visual Factors**

$$
\hat{x}_{u, i} = \alpha + \beta_u + \beta_i + \gamma^T_u \gamma_i + \theta^T_u \theta_i
$$

The inner product $ \theta^T_u \theta_i$ models the visual interaction between $u$ and $i$, i.e., the extent to which the user $u$ is attracted to each of $D$ visual dimensions.  

One naive way to implement the model would be directly user Deep CNN features $f_i$ of item $i$  as $\theta_i$ in the above equation.  
$\rightarrow$ However, this would present issues due to the high dimensionality of the features.  
$\rightarrow$ Instead, propose to learn an embedding kernel which linearly transforms such high-dimensional features into a much lower-dimensional "visual rating" space:  

$$
\theta_i = \mathbf{E}f_i
$$

$\mathbf{E}$ is a $D \times F$ matrix embedding Deep CNN feature space($F$-dimensional) into visual space($D$-dimensional), where $f_i$ is the original visual feature vector for itme $i$.  
$\rightarrow$ This embedding is efficient in the sense that all items share the same embedding matrix which significantly reduces the number of parameters to learn.

---
### **2.3 Add Visual Bias Term**

$$
\hat{x}_{u, i} = \alpha + \beta_u + \beta_i + \gamma^T_u \gamma_i + \theta^T_u(\mathbf{E}f_i) + \beta^{\prime T}f_i
$$

Introduce a visual bias term $\beta^\prime$ whose inner product with $f_i$ mmodels users' overall opinion toward the visual appearance of a given item.  
Final prediction model is:  

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/vbpr_figure1.PNG?raw=true" width=400>
</p>

---
# **3. Model Learning Using BPR**  

Bayesian Personalized Ranking (BPR) is a pairwise ranking
optimization framework which adopts stochastic gradient
ascent as the training procedure. (See paper review [here](https://ddoeunn.github.io/2020/11/06/PAPER-REVIEW-BPR-Bayesian-personalized-ranking-from-implicit-feedback.html).)  

* A training set $D_s$ consists of triples of the form $(u, i, j)$, where $u$ denotes the user together with an item $i$ about which they expressed positive feedback, and a non-observed item $j$

$$
D_s = \{(u, i, j) \vert u \in \mathcal{U} \wedge i \in \mathcal{I}^{+}_u \wedge j \in \mathcal{I} \backslash \mathcal{I}^{+}_u \}
$$  

* The following optimization criterion is used for personalized ranking(BPR-OPT) where $\sigma$ is the sigmoid sunction and $\lambda_\Theta$ is a model specific recgularization hyperparameter.


$$
\underset{(u, i, j) \in D_s}{\sum} \ln \sigma(\hat{x}_{uij}) - \lambda_{\Theta} \lVert \Theta \rVert^2
$$


* When using Matrix Factorization as the preference predictor(BPR-MF), $\hat{x}_{uij}$ is defined as follow where $\hat{x}_{u, i}$ and $\hat{x}_{u, j}$ are difine by equation $\hat{x}_{u, i} = \alpha + \beta_u + \beta_i + \gamma^T_u \gamma_i + \theta^T_u(\mathbf{E}f_i) + \beta^{\prime T}f_i$

$$
\hat{x}_{uij} = \hat{x}_{u, i} - \hat{x}_{u, j}
$$

* Update Rule  
The learning algorithm updates parameters in the following fashion where the model parameters $\Theta$ are $\beta_u$, $\beta_i$, $\gamma_u$, $\gamma_i$ ; non-visual parameters and $\theta_u$, $\beta^\prime$, $\mathbf{E}$ ; visual parameters

$$
\Theta \leftarrow \Theta + \eta \cdot (\sigma (- \hat{x}_{uij}) \frac{\partial \hat{x}_{uij}}{\partial \Theta} - \lambda_\Theta \Theta)
$$

$$
\theta_u \leftarrow \theta_u + \eta \cdot (\sigma (- \hat{x}_{uij}) \mathbf{E} (f_i - f_j) - \lambda_\Theta \theta_u)
$$

$$
\beta^\prime \leftarrow \beta^\prime + \eta \cdot (\sigma (-\hat{x}_{uij})(f_i - f_j) -\lambda_\beta \beta^\prime)
$$

$$
\mathbf{E} \leftarrow \mathbf{E} + \eta \cdot (\sigma(-\hat{x}_{uij}) \theta_u (f_i - f_j)^T -\lambda_\mathbf{E}\mathbf{E})
$$


---
# **4. Experiments**  
### **4.1 Datasets**
#### Amazon.com
* Users' review histories as implicit feedback
* Women's and Men's Clothing  
:visual features have already been demonstrated to be meaningful
* Cell Phones & Accessories  
:expect visual characteristics to play a smaller but possibly still significant role.

#### Tradesy.com
* users' purchase histories and "thumbs-up"
* visual information

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/vbpr_table2.PNG?raw=true" width=400>
</p>


---
### **4.2 Visual Features**
* For each item $i$ in the above datasets, collect one product image.
* Extract visual features $f_i$ using the Caffe reference model [(Jia et al., 2014)](http://proceedings.mlr.press/v32/donahue14.pdf), which implements the CNN architecture proposed by [Krizhevsky, Sutskever, and Hinton (2012).](https://dl.acm.org/doi/pdf/10.1145/3065386)
* The architecture has 5 convolutional layers followed by 3 fully-connected layers, and has been pre-trained on 1.2 million ImageNet (ILSVRC2010) images.
* Take the output of the second fully-connected layer (i.e. FC7), to obtain an $F = 4096$ dimensional visual feature vector $f_i$.

---
### **4.3 Evaluation Methodology**
* Split data into training / validation / test sets by selecting for each user $u$ a random item to be used for validation $\mathcal{V}_u$ and another for testing $\mathcal{T}_u$. (all remaining data is used for training)
* Metric

$$
AUC = \frac{1}{\lvert \mathcal{U} \rvert}\underset{u}{\sum}\frac{1}{\lvert E(u) \rvert}\underset{(i, j) \in E(u)}{\sum}\delta(\hat{x}_{u, i} > \hat{x}_{u, j})\\
\text{where the set of evaluation pairs for user }u \text{ is defined as }\\
E(u) = \{ (i, j) \vert (u, i) \in \mathcal{T} \wedge (u, j) \notin (\mathcal{P}_u \cup \mathcal{V}_u \cup \mathcal{T}_u) \}\\
\text{and }\delta(b) \text{ is an indicator function}
$$

---
### **4.4 Baselines**
* Random(RAND)  
:ranks items randomly for all users
* MostPopular(MP)  
:ranks items according to their popularity and is non-personalized.
* [MM-MF](https://dl.acm.org/doi/pdf/10.1145/2043932.2043989?casa_token=bEh7Pf9tkegAAAAA:TtjFG2UYxXFU0gel-LXJGsjyMtVKpX9XLn4m1c3JevxWbTEaeZ2Q14YGDYg102hMn1jvq7LC0WtwKFc)  
:pairwise MF model which is optimized for a hinge ranking loss on $x_{uij}$ and trained using SGA as in BPR-MF
* [BPR-MF](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf)  
:pairwise method personalized ranking for implicit feedback datasets.
* [Image-based Recommendation(IBR)](https://dl.acm.org/doi/pdf/10.1145/2766462.2767755?casa_token=VjmNt5dw7wsAAAAA:Sa1U1nc8HVmJCpI3KB4dEuo8PUU35NDU0fk5UkbY9lZtgoUi5DqHIEi4s3WNvVRyAMj77Z-giKpd7Es)  
:learns a visual space and retrieves stylistically similar items to a query image. prediction is then performed by nearest0neighbor search in the learned visual space.

---
### **4.5 Performance**  

<p align="center">
<img src="https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/vbpr_table3.PNG?raw=true" width=600>
</p>

* By combining the strengths of both MF and content-based methods, VBPR outperforms all baselines in most cases.
* VBPR exhibits particularly large improvemnets on Tradesy.com dataset, since it is an inherently cold start datset due to the 'one-off' nature of trades
* Visual features show greater benefits on clothing than cellphone datasets. Presumably this is because visual factors play a smaller (though still significant) role when selecting cellphones as compared to clothing.


---
# **5. Summary**
1. This paper investigated the usefulness of visual features for personalized ranking tasks on implicit feedback datasets.
2. It proposed a scalable method that incorporates visual features extracted from product images into Matrix Factorization, in order to uncover the "visual dimensions" that most influence people's behavior.
3. Experimental results on multiple large real-world datasets demonstrate that
    + VBPR can significantly outperform state-of-the-art ranking techniques
    + VBPR can alleviate cold start issues.

---
### **Reference**
[[1]](https://arxiv.org/pdf/1510.01784.pdf) He, Ruining, and Julian McAuley. "VBPR: visual bayesian personalized ranking from implicit feedback." arXiv preprint arXiv:1510.01784 (2015).  
[[2]](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf) Rendle, Steffen, et al. "BPR: Bayesian personalized ranking from implicit feedback." arXiv preprint arXiv:1205.2618 (2012).  
[[3]](http://proceedings.mlr.press/v32/donahue14.pdf) Donahue, Jeff, et al. "Decaf: A deep convolutional activation feature for generic visual recognition." International conference on machine learning. 2014.  
[[4]](https://dl.acm.org/doi/pdf/10.1145/2766462.2767755?casa_token=VjmNt5dw7wsAAAAA:Sa1U1nc8HVmJCpI3KB4dEuo8PUU35NDU0fk5UkbY9lZtgoUi5DqHIEi4s3WNvVRyAMj77Z-giKpd7Es)   McAuley, Julian, et al. "Image-based recommendations on styles and substitutes." Proceedings of the 38th international ACM SIGIR conference on research and development in information retrieval. 2015.
