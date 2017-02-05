Title: Intro to Machine Learning
Category: Data Science
Tags: machinelearning, sklearn, scikit-learn
Slug: machine_learning_basic
Authors: Kimi Yuan
Status: draft

[TOC]

# Overview

## What's Machine Learning

Within the field of *data analytics*, machine learning is a method used to devise complex models and algorithms that lend themselves to **prediction**; in commercial use, this is known as predictive analytics.

* **Supervised learning**: The computer is presented with example inputs and their desired outputs, given by a "teacher", and the goal is to learn a general rule that maps inputs to outputs.
* **Unsupervised learning**: No labels are given to the learning algorithm, leaving it on its own to find structure in its input. Unsupervised learning can be a goal in itself (discovering hidden patterns in data) or a means towards an end (feature learning).
* **Reinforcement learning**: A computer program interacts with a dynamic environment in which it must perform a certain goal (such as driving a vehicle), without a teacher explicitly telling it whether it has come close to its goal. Another example is learning to play a game by playing against an opponent.

## Choose Your Own Algorithm

### Choose the right estimator

http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html

![scikit-learn algorithm cheat-sheet]({filename}/images/ml_map.jpg)

### Process for ML

1. Do some research
2. Find sklearn documentation
3. Deploy it
4. Use it to make predictions
5. Evaluate it


# Supervised Learning

## Naive Bayes

### Bayes' Theorem

Bayes' theorem is stated mathematically as the following equation:
$$
P(A \mid B) = \frac{P(B \mid A)\ P(A)}{P(B)}
$$
where A and B are events and P(B) ≠ 0.

P(A) and P(B) are the probabilities of observing A and B without regard to each other.

P(A) is the *prior probablity*.
P(A | B), the *posterior probability*, is the probability of observing event A given that B is true.
P(B | A) , the *likelihood*, is the probability of observing event B given that A is true.

**Posterior is proportional to prior times likelihood**.

### Naive Bayes

Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of **independence** between every pair of features. Given a class variable *y* and a dependent feature vector $x_1$ through $x_n$, Bayes’ Theorem states the following relationship:
$$
P(y \mid x_1, ..., x_n) = \frac{P(y)P(x_1, ...,x_n \mid y)}{P(x_1, ...,x_n)}
$$
Using the naive independence assumption that
$$
P(x_i \mid y,x_1,...,x_{i+1}, ..., x_n) = P(x_i \mid y)
$$
for all *i*, this relationship is simlified to
$$
P(y \mid x_1, ..., x_n) = \frac{P(y)\prod_{i=1}^n{P(x_i \mid y)}}{P(x_1,...x_n)}
$$
Since $P(x_1, …, x_n)$ is constant given the input, we can use the following classification rule:
$$
P(y \mid x_1, ..., x_n)  \propto P(y)\prod_{i=1}^n{P(x_i \mid y)}
$$
and we can use Maximum A Posteriori (MAP) estimation to estimate P(y) and $P(x_i \mid y)$; the former is then the relative frequency of class y in the training set.


The different naive Bayes classifiers differ mainly by the assumptions they make regarding the distribution of $P(x_i \mid y)$.

* Gaussian Naive Bayes
* Multinomial Naive Bayes
* Bernoulli Naive Bayes

### Pro and Con

Naive Bayes classifiers have worked quite well in many complex real-world situations, famously document classification and spam filtering.

An advantage of naive Bayes is that it only requires a small number of training data to estimate the parameters necessary for classification.

Although naive Bayes is known as a decent classifier, it is known to be a bad estimator, so the probability outputs from `predict_proba` are not to be taken too seriously.

Phrases that encompass multiple words and have distinctive meanings don't work really well in Naive Bayes, e.g., Chicago Bulls.

### Code Example



```python
>>> from sklearn import datasets
>>> iris = datasets.load_iris()
>>> from sklearn.naive_bayes import GaussianNB
>>> gnb = GaussianNB()
>>> y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
>>> print("Number of mislabeled points out of a total %d points : %d"
...       % (iris.data.shape[0],(iris.target != y_pred).sum()))
Number of mislabeled points out of a total 150 points : 6
```



## Support Vector Machines (SVM)

### Parameters in SVM:

Arguments passed when you create your classifier before fitting, **huge difference** will have.

* kernel
* gamma
* C


### Kernel Trick

C - controls tradeoff between **smooth decision boundary** and **classifying training points correctly**.  Larger C means more training data correct.

The 'gamma' parameter actually has no effect on the 'linear' kernel for SVMs. The key parameter for this kernel function is "C",

Overfitting



### Pro and Con:

Work well in a clear margin of separation

Not perform well in very large data sets

Not work well with lots of noise



## Decision Trees

Parameter



min_samples_split: Avoid overfitting

### Entropy

controls how a DT decides where to split the data.

Definition: measure of **impurity** in a bunch of examples

Formula:


$$
entropy = \sum_{i}-(P_i)log_2(P_i)
$$


entropy = 0 (minimum): All examples are same class.

entropy = 1 (maximum): All examples are evenly split between classes.



### Information Gain

DT algorithm: maximize information gain

Bias-Variance Dilemma?





Homework

* K nearest neighbors
* adaboost
* random forest






## Comparing Classification & Regression



| Property                   | Supervised classification | Regression                  |
| :------------------------- | :------------------------ | :-------------------------- |
| Output type                | discrete (class labels)   | continuous (number)         |
| What are u trying to find? | decision boundary         | best fit line               |
| Evaluation                 | accuracy                  | sum of squared error, or r2 |



## Linear Regression



### Linear Regression  Error

error = actual value - predicted value

The best regression is the one that minimizes the sum of squared errors:
$$
\sum_{all\_training\_points}(actual - predicted)^2
$$

actual —> training points,

predicted —> predictions from regression $y = mx + b$



Several algorithms:

* Ordinary Least Squares (OLS)
* Gradient Descent

Problem of Sum of Squared Errors will grow larger when the amount of points in data set increases.



r^2^ ( r squared) of a regression

r^2^ answers the question how much of my change in the output(y) explained by the change in my input (x).

0.0 < r^2^ < 1.0

Closing to 0 means line isn't doing a good job of capturing trend in data set.

Closing to 1 means line does a good job of describing relationship between input & output.

It's irrelative to the amount of data set. It's better than sum of squared error.



### Code It Up

```python
from sklearn.linear_model import LinearRegression

ref = LinearRegression()
reg.fit(X, y)

reg.predict(x1) # Prediction on x1
reg.score(X, y) # r-squared score
reg.coef_       # slope
reg.intercept   # intercept
```



### Multivariate Regression

TBD

one-vs-all/one-vs-rest

## Outliers

What causes outliers?

* sensor malfunction
* data entry errors
* freak event



Outlier Detection/Rejection

1. Train
2. Remove points with largest residual error (~10%)
3. Train again
4. (Optional) repeat step 2 and 3




A **statistical error** (or disturbance) is the amount by which an observation differs from its expected value.

A **residual error** (or fitting deviation), on the other hand, is an observable estimate of the unobservable statistical error.



Consider the example with men's heights and suppose we have a random sample of n people. The sample mean could serve as a good estimator of the population mean. Then we have:

* The difference between the height of each man in the sample and the unobservable population mean is a statistical error, whereas
* The difference between the height of each man in the sample and the observable sample mean is a residual error.




# Unsupervised Learning



## Clustering

*Clustering* is the task of partitioning the dataset into groups, called clusters. The goal is to split up the data in such a way that points.

### K-Means

k-means clustering is one of the simplest and most commonly used clustering algorithms. It tries to find *cluster centers* that are representative of certain regions of the data.

Procedure:

1. ***Initialization***. Declaring k data points randomly as cluster centers.
2. **Assign Points**. Assign data point to the closet cluster center.
3. **Recompute Centers**. Cluster centers are updated to be the mean of the assign points.
4. Repeat 2-3 until the assignment of points to cluster centers remained unchanged.



K-Means Cluster Visualization
You can play with k-means clustering yourself here: http://www.naftaliharris.com/blog/visualizing-k-means-clustering/

### Agglomerative testing





### DBSCAN





# Feature Scaling & Feature Selection

Formula



## Feature Selection

Adding more features makes all models more complex, and so increases the chance of overfitting. When adding new features, or with high-dimensional datasets in general, it can be a good idea to reduce the number of features to only the most useful ones, and discard the rest. This can lead to simpler models that generalize better.

### Univariate Statistics



### Model-Based Feature Seclection



### Iterative Feature Selection



# Dimensionality Reduction



## Principle Component Analysis - PCA



Projection onto direction of maximal variance *minimizes distance* from old (higher dimensional) data point to its new transformed value and minimizes **information loss**.

* Systematical way to transform input features into principal components.
* Use principal components as new features.
* Principal components are directions in data that maximize variance (minimize information loss) when you project/compress down onto them.
* More variance of data along a PC, higher that PC is ranked.
* Most variance/most information —> first PC, second most variance (without overlapping first PC) —> second PC
* max number of PCs = number of input features

### When To Use PCA

* Find out latent features driving the patterns in data
* Dimensionality reduction
  * Visualize high dimensional data
  * Reduce noise
  * Make other algorithms (regression, classification) work better with fewer inputs




# Text Learning

Bag of Words

Vocabulary: Not all words are equal.

Stopwords: low information, highly frequent word - [the, in, for, you, will, have, be]

NLTK

Stemming

```python
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
stemmer.stem("responsiveness")
stemmer.stem("responsivity")
```

You want to form word stems before constructing your bag-of-words.

Tf-Idf: Term frequency–Inverse document frequency, weighting rare words higher

# Model Validation & Evaluation

### Train/Test Split in sklearn



## Cross Validation

### K-Fold Cross Validation

![kfolder](/Users/Kimi/Documents/Developer/Web_Projects/pelican_tech_blog/content/images/kfolder.png)

### Stratified K-Fold Cross Validation

![stratified_kfold](/Users/Kimi/Documents/Developer/Web_Projects/pelican_tech_blog/content/images/stratified_kfold.png)

`cross_val_score` uses stratified k-fold cross-validation by default for classification and k-fold cross-validation for regression.

### Leave-one-out Cross Validation

http://scikit-learn.org/stable/modules/cross_validation.html

## Evaluation Metrics

### Confusion Matrix

|                         | Spam                | not Spam            |
| ----------------------- | ------------------- | ------------------- |
| predict "Spam" (P)      | True Positive (TP)  | False Positive (FP) |
| predict "Not Spam"  (N) | False Negative (FN) | True Negative (TN)  |

### Accuracy (ACC)

The accuracy should actually be no. of all data points labeled correctly divided by all data points.
$$
ACC = \frac{TP + TN}{P + N}
$$


### Precision and Recall

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

### F1 score

In statistical analysis of binary classification, the F1 score (also F-score or F-measure) is a measure of a test's accuracy. It considers both the precision and the recall of the test to compute the score: The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst at 0.

The traditional F-measure or balanced F-score (F1 score) is the *harmonic mean* of precision and recall:
$$
F_1 = 2 \cdot \frac{precision\cdot recall}{precision+recall} = \frac{2\ TP}{2TP + FP + FN}
$$

More terminologies can be found in https://en.wikipedia.org/wiki/Confusion_matrix .



GD
