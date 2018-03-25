Title: Python Snippets
Category: Python
Tags: python, numpy, pandas, ipython, jupyter
Slug: python_snippets
Authors: Kimi Yuan
Summary: Useful Python Snippets in both standard and 3rd party libraries.

[TOC]

# Standard Libraries

## Dictionary

Dicts can be used to emulate switch/case statements.

```python
def dispatch_if(operator, x, y):
    if operator == 'add':
        return x + y
    elif operator == 'sub':
        return x - y
    elif operator == 'mul':
        return x * y
    elif operator == 'div':
        return x / y
    else:
        return None


def dispatch_dict(operator, x, y):
    return {
        'add': lambda: x + y,
        'sub': lambda: x - y,
        'mul': lambda: x * y,
        'div': lambda: x / y,
    }.get(operator, lambda: None)()

>>> dispatch_if('mul', 2, 8)
16

>>> dispatch_dict('mul', 2, 8)
16

>>> dispatch_if('unknown', 2, 8)
None

>>> dispatch_dict('unknown', 2, 8)
None
```



## ipaddress package

It becomes Python standard library since 3.3. If using Python version before 3.3,  run 'pip install ipaddress'.

```python
In [190]: import ipaddress
# It run on Python 2.7, but no need of decode() for Python 3
In [191]: addr = ipaddress.IPv4Network('10.0.0.0/8'.decode()) 

In [192]: addr.hostmask
Out[192]: IPv4Address(u'0.255.255.255')

In [193]: str(addr.hostmask)
Out[193]: '0.255.255.255'

In [194]: addr.netmask
Out[194]: IPv4Address(u'255.0.0.0')
```



## Creating Classes with `type`

You can use `type` to determine the type of an object, but you can also provide the name, parents, and attributes map, and it will return a class. [1]

```Python
>>> def howl(self):
...     return "HOWL"

>>> parents = ()
>>> attrs_map = {'speak': howl}
>>> F = type('F', parents, attrs_map)

>>> f = F()
>>> print(f.speak())
HOWL
```

## sys

### Show the current Python version

```python
In [1]: import sys

In [2]: sys.version
Out[2]: '3.6.1 |Anaconda 4.4.0 (x86_64)| (default, May 11 2017, 13:04:09) \n[GCC 4.2.1 Compatible Apple LLVM 6.0 (clang-600.0.57)]'

In [3]: sys.version_info
Out[3]: sys.version_info(major=3, minor=6, micro=1, releaselevel='final', serial=0)
```

## functools

```python
# import module
import functools
```

### partial

Partial pre-filling a couple of arguments before they're called in a function. [2]

```python
def adder(x, y): 
  return x + y 

# it adds! 
assert adder(1, 1) == 2
assert adder(5, 5) == 10
assert adder(6, 2) == 8

# pre fill y with the value 5  
add_five = functools.partial(adder, y=5)

#now it adds 5!
# x=1, y=5 
assert add_five(1) == 6
# x=5, y=5
assert add_five(5) == 10
# x=2, y=5
assert add_five(2) == 7
```

## collections

### defaultdict

Find all Key-Elements by the same Value in Dicts.

```python
from collections import defaultdict
some_dict = { 'abc':'a', 'cdf':'b', 'gh':'a', 'fh':'g', 'hfz':'g' }
new_dict = defaultdict(list)
for k, v in some_dict.items():
    new_dict[v].append(k)
```



### OrderDict





### namedtuple

Using namedtuple is way shorter than defining a class manually:

```python
>>> from collections import namedtuple
>>> Car = namedtup1e('Car' , 'color mileage') # could be 'color, mileage' or ['color', 'mileage']

# Our new "Car" class works as expected:
>>> my_car = Car('red', 3812.4)
>>> my_car.color
'red'
>>> my_car.mileage
3812.4

# We get a nice string repr for free:
>>> my_car
Car(color='red' , mileage=3812.4)

# Like tuples, namedtuples are immutable:
>>> my_car.color = 'blue'
AttributeError: "can't set attribute"

# Easy to convert to OrderedDict    
>>>my_car._asdict()
OrderedDict([('color', 'red'), ('mileage', 3812.4)])
```



## itertools

### Combinations and Permutations

```python
In [301]: from itertools import permutations
In [302]: [p for p in permutations('abc')]
Out[302]:
[('a', 'b', 'c'),
 ('a', 'c', 'b'),
 ('b', 'a', 'c'),
 ('b', 'c', 'a'),
 ('c', 'a', 'b'),
 ('c', 'b', 'a')]

In [303]: [p for p in permutations('abc', 2)]
Out[303]: [('a', 'b'), ('a', 'c'), 
           ('b', 'a'), ('b', 'c'),
           ('c', 'a'), ('c', 'b')]

In [305]: from itertools import combinations
In [306]: [c for c in combinations('abc',3)]
Out[306]: [('a', 'b', 'c')]

In [307]: [c for c in combinations('abc',2)]
Out[307]: [('a', 'b'), ('a', 'c'), ('b', 'c')]

In [308]: [c for c in combinations('abc',1)]
Out[308]: [('a',), ('b',), ('c',)]
    
    
# Allow the same item to be chosen more than once
In [309]: from itertools import combinations_with_replacement
In [310]: [c for c in combinations_with_replacement('abc',3)]
Out[310]:
[('a', 'a', 'a'),
 ('a', 'a', 'b'),
 ('a', 'a', 'c'),
 ('a', 'b', 'b'),
 ('a', 'b', 'c'),
 ('a', 'c', 'c'),
 ('b', 'b', 'b'),
 ('b', 'b', 'c'),
 ('b', 'c', 'c'),
 ('c', 'c', 'c')]
```



### zip_longest

```python
>>> a = [1, 2, 3]
>>> b = ['w', 'x', 'y', 'z'] 
>>> for i in zip(a,b):
	... print(i)

(1, 'w')
(2, 'x')
(3, 'y')

>>> from itertools import zip_longest 
>>> for i in zip_longest(a,b):
    ... print(i)

(1, 'w')
(2, 'x')
(3, 'y')
(None, 'z')

>>> for i in zip_longest(a, b, fillvalue=0):
    ... print(i)
    
(1, 'w')
(2, 'x')
(3, 'y')
(0, 'z')
>>>
```



### islice

Taking slices of iterators and generators.

```python
>>> def count(n):
... 	while True:
...			yield n
... 		n += 1

>>> import itertools
>>> for x in itertools.islice(c, 10, 20): ... print(x)
...
10
11
12
13
14
15
16
17
18
19
>>>
```

### dropwhile

The returned iterator discards the first items in the sequence as long as the
supplied function returns True. Afterward, the entirety of the sequence is produced.

```python
>>> from itertools import dropwhile
>>> with open('/etc/passwd') as f:
... for line in dropwhile(lambda line: line.startswith('#'), f): ... print(line, end='')
```



## math

### Round, Ceil and Floor

```python
In [45]: round(3.3333, 2)
Out[45]: 3.33

In [46]: round(6.666, 2)
Out[46]: 6.67

In [47]: import math

In [48]: math.ceil(3.333)
Out[48]: 4
   
In [49]: math.floor(3.9)
Out[49]: 3
```



### hypot

**math.hypot(x, y)** returns the Euclidean norm, `sqrt(x*x + y*y)`. This is the length of the vector from the origin to point `(x, y)`.





## tarfile

```python
with tarfile.open("somefile_pdc.tar.gz", "r:gz") as f:
    f.extract("a.txt")
    
    # or
    f.extract_all()
```



## textwrap

Collapse and truncate the given *text* to fit in the given *width*.

First the whitespace in *text* is collapsed (all whitespace is replaced by single spaces). If the result fits in the *width*, it is returned. Otherwise, enough words are dropped from the end so that the remaining words plus the `placeholder` fit within `width`

```python
>>> textwrap.shorten("Hello world", width=10, placeholder="...")
'Hello...'
```



## pickle

```python
import pickle
pickle.dump(food, open('food.pkl', 'wb'))

# load a pickle file from python2
food = pickle.load(open("food.pkl", 'rb'), encoding='latin1')
```



# Numpy & Scipy

```python
# import modules
import numpy as np
```

## Reshape Numpy Arrays

One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions. [3]

```python
In [12]: np.arange(8)
Out[12]: array([0, 1, 2, 3, 4, 5, 6, 7])

In [13]: _.shape
Out[13]: (8,)

In [14]: np.arange(8).reshape(2,4)
Out[14]:
array([[0, 1, 2, 3],
       [4, 5, 6, 7]])

In [15]: _.shape
Out[15]: (2, 4)

In [16]: np.arange(8).reshape(2,-1) # the unspecified value is inferred to be 4
Out[16]:
array([[0, 1, 2, 3],
       [4, 5, 6, 7]])

In [17]: _.shape
Out[17]: (2, 4)
```



###Distance Computations

```python
# L2 distance between 2 vectors
dist = np.linalg.norm(vect_a - vect_b)
# or
dist = np.sqrt(np.sum(vect_a - vect_b)**2)


# L2 distances between vector (m,) and matrix (n,m), output (n,)
dists = np.sqrt(np.sum((matrix - np.tile(vector, [n,1]))**2,1))
# or
dists = np.linalg.norm(matrix - np.tile(vector, [n,1]), axis=1)
# or output (n,1)
from scipy.spatial.distance import cdist
dists = cdist(matrix, vector.reshape(1,-1))

# L2 distances between matrix_a (m, k) and matrix_b (n, k), output (m, n)
dists = cdist(matrix_a, matrix_b)
```





# Pandas

```python
# import modules
import pandas as pd
```

## Read Excel file

```python
# Import the excel file and call it xl
In [111]: xl = pd.ExcelFile('example.xls')
    
In [112]: xl
Out[112]: <pandas.io.excel.ExcelFile at 0x106c62be0>

# View the sheet names in excel file
In [112]: xl.sheet_names
Out[112]:
['Sheet1']

# Load the xls file's Sheet1 as a dataframe
In [113]: df = xl.parse('Sheet1')
  
# Or load into DataFrame directly
In [114]: df = pd.read_excel('example.xls', sheetname='Sheet1')
```

## Read CSV file

```python
dataframe = pd.read_csv('nodePdcJob.log',header=2, delimiter="|") # skip 2 lines in the file header
```



# IPython & Notebook

## Display an image in Notebook 

```python
from IPython.display import display, Image
display(Image(filename="cat.jpg"))
```



### Auto-reload module

For IPython version 3.1, 4.x, and 5.x

```python
In [1]: %load_ext autoreload
In [2]: %autoreload 2
```



# References

\[1\]  [Tiny-Python-3.6-Notebook](https://github.com/mattharrison/Tiny-Python-3.6-Notebook/blob/master/python.rst)

[2\]  [Cleaner Code Through Partial Function Application](http://chriskiehl.com/article/Cleaner-coding-through-partially-applied-functions/)

[3\] [numpy.reshape —Numpy Manual](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html)

[4\] [Autoreload of modules in IPython](https://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython)

