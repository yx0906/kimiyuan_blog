Title: Word2Vec
Category: Data Science
Tags: NLP, machinelearning
Slug: word2vec
Authors: Kimi Yuan
Summary: 
Status: draft

[TOC]

# Word Representations

### Local/discrete representations

* **N-grams**
* **Bag-of-words**
* **One-hot vector**: Represent every word as an R|V|×1 vector with all 0s and one 1 at the index of that word in the sorted english language. In this notation, |V| is the size of our vocabulary.



![one_hot]({filename}/images/one_hot.jpeg)



### Continuous representations

Latent Semantic Analysis

Latent Dirichlet Allocation

Distributed Representations

[Vector space models](https://en.wikipedia.org/wiki/Vector_space_model) (VSMs) represent (embed) words in a continuous vector space where semantically similar words are mapped to nearby points ('are embedded nearby each other'). VSMs have a long, rich history in NLP, but all methods depend in some way or another on the [Distributional Hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics#Distributional_Hypothesis), which states that words that appear in the same contexts share semantic meaning. 

# Word2Vec 





![word2vec_models]({filename}/images/word2vec_models.jpeg)



## Skip-Gram model

Predicts the surrounding words in a window of length m given the current word.

Objective function: Maximize the log probability of any context word given the current center word:

$J(\theta) = \frac{1}{T}\displaystyle\sum_{t=1}^{T}\displaystyle\sum_{-m\leq j\leq m, j\neq 0}\log p(w_{t+j}|w_t)$

$p(o|c) = \cfrac{exp(u_o^Tv_c)}{\sum_{w=1}^Wexp(u_w^Tv_c)}$



![skip_gram]({filename}/images/skip_gram.jpeg)

**Notation for Skip-Gram Model:**

* $w_i$ : Word $i$ from vocabulary $V$
*  $\mathcal{V} ∈ \rm I\!R^{ n×|V|}$: Input word matrix
*  $v_i : i$-th column of $\mathcal{V}$ , the input vectorrepresentation of word $w_i$
*  $\mathcal{U} ∈ \rm I\!R^{ n×|V|}$|: Output word matrix
* $u_i : i$-th row of  $\mathcal{U}$, the output vectorrepresentation of word $w_i$

**Steps:**

1. We generate our one hot input vector $x$
2. We get our embedded word vectors for the context $v_c = \mathcal{V}x$
3. Generate 2m score vectors, $u_{c−m}, . . . , u_{c−1}, u_{c+1}, . . . , u_{c+m}$ using $u = \mathcal{U}v_c$
4. Turn each of the scores into probabilities, $y =$  softmax$(u)$
5. We desire our probability vector generated to match the true prob-abilities which is $y^{(c−m)}, . . . , y^{(c−1)}, y^{(c+1)}, . . . , y^{(c+m)}$, the one hotvectors of the actual output.



## Continuous Bag-of-Words model (CBOW) 

The CBOW architecture predicts the current word based on the context, and the Skip-gram predicts surrounding words given the current word.





## Software

### tensorflow



### word2vec







Reference
https://www.tensorflow.org/tutorials/word2vec