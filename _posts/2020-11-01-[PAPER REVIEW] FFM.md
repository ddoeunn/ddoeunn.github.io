---
layout: post
title: PAPER REVIEW_Field-aware Factorization Machines for CTR prediction
tags: [paper review, recommender system, matrix factorization, factorization machine]
use_math: true
---

Click-through rate(CTR) prediction plays an important role in computational advertising. This [paper](https://dl.acm.org/doi/pdf/10.1145/2959100.2959134?casa_token=2HFKgPvmUnQAAAAA:74DUN0wTfUgZu92OPlmGQsIpTlPVqJv7Dzjspa_ZMVJZ-k5j4e-Cw7hPzKusLJNY30O7VG8TXvcXCgI) establishes Field-aware Factorization Machines(FFM) as an effective method for classifying large sparse data including CTR prediction. FFM is a variant of FM that utilizes information that features can be grouped into field for most CTR dataset.

---
# **1. CTR dataset**
For most CTR datasets, "features" can be grouped into "field". In
Table1: example, three features ESPN, Vogue, and NBC, belong to theeld Publisher, and the other three features Nike, Gucci, and Adidas, belong to the eld Advertiser. FFM is a variant of FM that utilizes this information.

<p align="center">
<img src=https://github.com/ddoeunn/ddoeunn.github.io/blob/main/assets/img/post%20img/ffm_table1.PNG?raw=true  alt="ctr example"  width="400">
</p>

---
# **2. FFM**
