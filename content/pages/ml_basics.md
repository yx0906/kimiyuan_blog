Title: Machine Learning Basics
Category: Data Science
Tags: machinelearning, 
Slug: machine_learning_basic
Authors: Kimi Yuan
Status: draft

[TOC]

# Definition

"A computer program is said to learn from experience *E* with respect to some class of tasks *T* and performance measure *P*, if its performance at tasks in *T*, as measured by *P*, improves with experience *E*."      — [Tom M. Mitchell](https://en.wikipedia.org/wiki/Tom_M._Mitchell)

A machine learning algorithm is capable of improving a computer program’s performance at some task via experience. 

## The Task, *T*

* Classification

* Classification with missing inputs

* Regression

* Transcription

* Machine translation

* Structured output

* Anomaly detection

* Synthesis and sampling

* Imputation of missing values

* Denoising

* Density estimation or probability mass function estimation

* etc...

  ​

## The Performance Measure, *P*

Usually we are interested in how well the machine learning algorithm performs on data that it has not seen before, since this determines how well it will work when deployed in the real world. We therefore evaluate these performance measures using a **test set** of data that is separate from the data used for training the machine learning system.

### Classification Performance Measure

#### Confusion Matrix

|                         | Spam                | not Spam            |
| ----------------------- | ------------------- | ------------------- |
| predict "Spam" (P)      | True Positive (TP)  | False Positive (FP) |
| predict "Not Spam"  (N) | False Negative (FN) | True Negative (TN)  |



#### Accuracy (ACC)

The accuracy should actually be no. of all data points labeled correctly divided by all data points.
$$
ACC = \frac{TP + TN}{P + N}
$$

#### Precision and Recall

Pecision and recall are defined as:
$$
Precision = \frac{TP}{TP + FP}
$$

$$
Recall = \frac{TP}{TP + FN}
$$

*Precision* is the number of correct positive results divided by the number of all positive results. Out of all the items labeled as positive, how many truly belong to the positive class.

*Recall* is the number of correct positive results divided by the number of positive results that should have been returned. Or simply, how many positive items were 'recalled' from the dataset.

Often, there is an inverse relationship between precision and recall, where it is possible to increase one at the cost of reducing the other. So precision and recall scores are not discussed in isolation.

#### F1 score

In statistical analysis of binary classification, the F1 score (also F-score or F-measure) is a measure of a test's accuracy. It considers both the precision and the recall of the test to compute the score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst at 0.

The traditional F-measure or balanced F-score (F1 score) is the *harmonic mean* of precision and recall:
$$
F_1 = 2 \cdot \frac{precision\cdot recall}{precision+recall} = \frac{2\ TP}{2TP + FP + FN}
$$




### Regression Performance Measure



## The Experience, *E*

Machine learning algorithms can be broadly categorized as **unsupervised** or **supervised** by what kind of experience they are allowed to have during the learning process.

Most of the learning algorithms can be understood as being allowed to experience an entire **dataset**. A dataset is a collection of many **examples**. Sometimes we will also call examples **data points**. An example is a collection of **features** that have been quantitatively measured from some object or event that we want the machine learning system to process. We typically represent an example as a vector x ∈ Rn where each entry xi of the vector is another feature.

**Supervised learning**: The computer is presented with example inputs and their desired outputs (**label** or **target**), given by "teacher", and the goal is to learn a general rule that maps inputs to outputs.

**Unsupervised learning**: No labels are given to the learning algorithm, leaving it on its own to find structure in its input. Unsupervised learning can be a goal in itself (discovering hidden patterns in data) or a means towards an end (feature learning).

**Reinforcement learning**: A computer program interacts with a dynamic environment in which it must perform a certain goal (such as driving a vehicle), without a teacher explicitly telling it whether it has come close to its goal. Another example is learning to play a game by playing against an opponent.

Some machine learning algorithms do not just experience a fixed dataset. For example, **reinforcement learning** algorithms interact with an environment, so there is a feedback loop between the learning system and its experiences.

Most machine learning algorithms simply experience a dataset.

One common way of describing a dataset is with a **design matrix**. A design matrix is a matrix containing a different example in each row. Each column of the matrix corresponds to a different feature.

​				

# Capacity, Overfitting and Underfitting

The central challenge in machine learning is that we must perform well on *new, previously unseen* inputs — not just those on which our model was trained. The ability to perform well on previously unoberved inputs is called **generalization**.		

The train and test data are generated by a probability distribution over datasets called the **data generating process**. We typically make a set of assumptions known collectively as the **independent, identically distributed (i.i.d.) assumptions**. These assumptions are that the examples in each dataset are **independent** from each other, and that the train set and test set are **identically distributed**, drawn from the same probability distribution as each other. The same distribution is then used to generate every train example and every test example. We call that distribution the **data generating distribution**, denoted $p_{data}$ . This probabilistic framework and the i.i.d. assumptions allow us to mathematically study the relationship between training error and test error.

Machine learning algorithms will generally perform best when their capacity
is appropriate for the true complexity of the task they need to perform and the
amount of training data they are provided with. 

* **Underfitting**: Models with insufficient capacity are unable to solve complex tasks.
* **Overfitting**: Models with high capacity can solve complex tasks, but when their capacity is higher than needed to solve the present task they may overfit.

![underfit_overfit]({filename}/images/underfit_overfit.jpg)

​				
**Occam’s razor** (c. 1287-1347): Among competing hypotheses that explain known observations equally well, one should choose the “simplest” one. 	

The most well-known of statistical learning theory to provide mean of quantifying model capacity is the **Vapnik-CHervonenkis dimension**.

The most important results in statistical learning theory show that the discrepancy between training error and generalization erroris bounded from above by a quantity that grows as the model capacity grows but shrinks as the number of training examples increases.

![capacity]({filename}/images/capacity.jpg)



The **no free lunch theorem** for machine learning (Wreomlpert, 1996) states that, averaged over all possible data generating distributions, every classification algorithm has the same error rate when classifying previously unobserved points. In other words, in some sense, no machine learning algorithm is universally any better than any other. The most sophisticated algorithm we can conceive of has the same average performance (over all possible tasks) as merely predicting that every point belongs to the same class.


**Parametric models** learn a function described by a parameter vector whose size is finite and fixed before any data is observed. Assumptions can greatly simplify the learning process, but can also limit what can be learned. 
**Non-parametric models** have no such limitation.
​		
​	


​				
​			
​		
​	

## Regularization		

​		
​	

## Maximum Likelihood Estimation



# References

[1] http://www.deeplearningbook.org Chapter 5, Machine Learning Basics from Deep Learning textbook

[2] https://en.wikipedia.org/wiki/Confusion_matrix 

[3] http://www.damienfrancois.be/blog/files/modelperfcheatsheet.pdf Model Performance Cheatsheet

[4] http://scikit-learn.org/stable/modules/model_evaluation.html

[5] http://machinelearningmastery.com/parametric-and-nonparametric-machine-learning-algorithms/ Parametric and Nonparametric Machine Learning Algorithms