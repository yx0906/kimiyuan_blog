Title: Nature Language Process
Category: Data Science
Tags: NLP, machinelearning, 
Slug: nlp
Authors: Kimi Yuan
Status: draft

[TOC]

# Models

## Bag of words





## word2vec

### skip-gram

### CBOW







# Text Learning

Bag of Words

Vocabulary: Not all words are equal.

Stopwords: low information, highly frequent word - [the, in, for, you, will, have, be]

NLTK

Stemming

\```python

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

stemmer.stem("responsiveness")

stemmer.stem("responsivity")

\```

You want to form word stems before constructing your bag-of-words.

Tf-Idf: Term frequency–Inverse document frequency, weighting rare words higher