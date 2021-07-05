---
layout: post
title: [Python] 추천시스템 Neural Collaborative Filtering 알고리즘 구현
tags: [recommender system, NCF, Collaborative Filtering, Matrix Factorization, Python]
use_math: true
---

***추천시스템 NCF 알고리즘 파이썬 구현***

* 논문 -> [He, Xiangnan, et al. "Neural collaborative filtering." Proceedings of the 26th international conference on world wide web. 2017.
](https://dl.acm.org/doi/pdf/10.1145/3038912.3052569?casa_token=FfITqllG5HMAAAAA:rI_bL7aiSwK9r061e8X7_SEIpBIfLd8_MGB3yMrIlj53dzlfvN97S_qZDIgKPepzSjjy5cFHEUgCgvY)

---
# **0. 문제 정의**
본 논문에서는 암시적 피드백 (Implicit Feedback)을 이용한 추천시스템을 다룹니다.

암시적 피드백은 시청기록, 구매내역처럼 사용자의 선호도를 간접적으로 파악할 수 있는 데이터입니다. 이와 상반되는 개념으로 명시적 피드백 (Explicit Feedback)이 있습니다. 명시적 피드백은 별점이나 좋아요/싫어요 버튼처럼 사용자가 직접적으로 자신의 선호를 나타낸 데이터 입니다.

명시적 피드백은 사용자가 직접 선호도를 나타내야 하기 때문에 데이터를 수집하기 어려운 반면에, 암시적 피드백은 사용자의 어떤 행동을 통해 데이터를 간접적으로 수집하기 때문에 상대적으로 쉽게 데이터를 수집할 수 있다는 장점이 있습니다. 하지만 사용자의 선호를 간접적으로만 파악할 수 있으며, 특히 직접적이고 확실한 negative feedback이 없기 때문에 본질적으로 noisy하다는 단점이 있습니다.






---
# **1. Matrix Factorizaiton이란?**





---
# **2. Matrix Factorization의 한계점**




---
# **3. **
