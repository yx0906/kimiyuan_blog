Title: Iterations and Comprehensions
Date: 2016-09-05 16:00
Modified: 2016-09-05 16:00
Category: Python
Tags: readingnotes, learningpython
Slug: learning-python-5
Authors: Kimi Yuan
Summary:

[TOC]

This is a reading note for CHAPTER 14 in the book **Learning Python, 5th Edition** by *Mark Lutz*.

This interface is most of what we call the **iteration** protocol in Python. Any object with a `__next__` method to advance to a next result, which raises StopIteration at the end of the series of results, is considered an iterator in Python. Any such object may also be stepped through with a for loop or other iteration tool, because all iteration tools normally work internally by calling `__next__`on each iteration and catching the StopIteration exception to determine when to exit.

Version skew note: In Python 2.X, the iteration method is named `X.next()` instead of `X.__next__()`. For portability, a `next(X)` built-in function is also available in both Python 3.X and 2.X (2.6 and later), and calls `X.__next__()` in 3.X and `X.next()` in 2.X. Apart from method names, iteration works the same in 2.X and 3.X in all other ways. In 2.6 and 2.7, simply use `X.next()` or `next(X)` for manual iterations instead of 3.X’s `X.__next__()`; prior to 2.6, use `X.next()` calls instead of `next(X)`.

## Manual Iteration: iter and next

Figure below sketches this full iteration protocol, used by every iteration tool in Python, and supported by a wide variety of object types. It’s really based on two objects, used in two distinct steps by iteration tools:
* The *iterable* object you request iteration for, whose `__iter__`is run by **iter**
* The *iterator* object returned by the iterable that actually produces values during the iteration, whose `__next__` is run by next and raises StopIteration when finished producing results.

![iteration.jpg]({filename}/images/iteration.jpg)

The Python iteration protocol, used by for loops, comprehensions, maps, and more, and supported by files, lists, dictionaries, generators, and more. Some objects are both iteration context and iterable object, such as generator expressions and 3.X’s flavors of some tools (such as map and zip). Some objects are both iterable and iterator, returning themselves for the iter() call, which is then a noop.

    :::python
    >>> L = [1, 2, 3]
    >>> I = iter(L)     # Obtain an iterator object from an iterable
    >>> I.__next__()    # Call iterator's next to advance to next item
    1
    >>> I.__next__()    # Or use I.next() in 2.X, next(I) in either line
    2
    >>> I.__next__()
    3
    >>> I.__next__()
    ...error text omitted...
    StopIteration


## List Comprehensions

#### Filter clauses: if
    :::python
    >>> lines = [line.rstrip() for line in open('script2.py') if line[0] == 'p'] >>> lines
    ['print(sys.path)', 'print(x ** 32)']

###Nested loops: for

    :::python
    >>> [x + y for x in 'abc' for y in 'lmn']
    ['al', 'am', 'an', 'bl', 'bm', 'bn', 'cl', 'cm', 'cn']


### Other Iteration Contexts
The **map** call used here briefly in the preceding chapter; it’s a built-in that applies a function call to each item in the passed-in iterable object. map is similar to a list comprehension but is more limited because it requires a function instead of an arbitrary expression. It also **returns** an iterable object itself in Python 3.X,

    :::python
    >>> map(str.upper, open('script2.py')) # map is itself an iterable in 3.X <map object at 0x00000000029476D8>
    >>> list(map(str.upper, open('script2.py')))
    ['IMPORT SYS\n', 'PRINT(SYS.PATH)\n', 'X = 2\n', 'PRINT(X ** 32)\n']


Many of Python’s other built-ins process iterables, too. For example, **sorted** sorts items in an iterable; zip combines items from iterables; **enumerate** pairs items in an iterable with relative positions; **filter** selects items for which a function is true; and **reduce** runs pairs of items in an iterable through a function. All of these *accept* iterables, and **zip, enumerate,** and **filter** also return an iterable in Python 3.X, like map.

    :::python
    >>> sorted(open('script2.py'))
    ['import sys\n', 'print(sys.path)\n', 'print(x ** 32)\n', 'x = 2\n']
    
    >>> list(zip(open('script2.py'), open('script2.py')))
    [('import sys\n', 'import sys\n'), ('print(sys.path)\n', 'print(sys.path)\n'), ('x = 2\n', 'x = 2\n'), ('print(x ** 32)\n', 'print(x ** 32)\n')]
    
    >>> list(enumerate(open('script2.py')))
    [(0, 'import sys\n'), (1, 'print(sys.path)\n'), (2, 'x = 2\n'), (3, 'print(x ** 32)\n')]
    
    >>> list(filter(bool, open('script2.py'))) # nonempty=True
    ['import sys\n', 'print(sys.path)\n', 'x = 2\n', 'print(x ** 32)\n']
    
    >>> import functools, operator
    >>> functools.reduce(operator.add, open('script2.py')) 'import sys\nprint(sys.path)\nx = 2\nprint(x ** 32)\n'

**Sorted** is a built-in that employs the iteration protocol—it’s like the original list sort method, but it returns the new sorted list as a result and runs on any iterable object. Notice that, unlike **map** and others, **sorted** returns an actual list in Python 3.X instead of an iterable.

*Everything* in Python’s built-in toolset that scans an object from left to right is defined to use the iteration protocol on the subject object. This even includes tools such as the **list** and **tuple** built-in functions (which build new objects from iterables), and the string join method (which makes a new string by putting a substring between strings contained in an iterable). Consequently, these will also work on an open file and automatically read one line at a time:

    :::python
    >>> list(open('script2.py'))
    ['import sys\n', 'print(sys.path)\n', 'x = 2\n', 'print(x ** 32)\n']
    
    >>> tuple(open('script2.py'))
    ('import sys\n', 'print(sys.path)\n', 'x = 2\n', 'print(x ** 32)\n')
    
    >>> '&&'.join(open('script2.py'))
    'import sys\n&&print(sys.path)\n&&x = 2\n&&print(x ** 32)\n'


Sequence assignment, the in membership test, slice assignment, and the list’s **extend** method also leverage the iteration protocol to scan, and thus read a file by lines automatically.

    :::python
    >>> a, b, c, d = open('script2.py')   # Sequence assignment
    >>> a, d
    ('import sys\n', 'print(x ** 32)\n')
    
    >>> a, *b = open('script2.py')        # 3.X extended form
    >>> a, b
    ('import sys\n', ['print(sys.path)\n', 'x = 2\n', 'print(x ** 32)\n'])
    
    >>> 'y = 2\n' in open('script2.py')   # Membership test
    False
    >>> 'x = 2\n' in open('script2.py')
    True
    
    >>> L = [11, 22, 33, 44]
    >>> L[1:3] = open('script2.py')
    >>> L
    [11, 'import sys\n', 'print(sys.path)\n', 'x = 2\n', 'print(x ** 32)\n', 44]
    
    >>> L = [11]
    >>> L.extend(open('script2.py'))      # list.extend method
    >>> L
    [11, 'import sys\n', 'print(sys.path)\n', 'x = 2\n', 'print(x ** 32)\n']

**Extend** iterates automatically, but **append** does not use the latter (or similar) to add an iterable to a list without iterating, with the potential to be iterated across later:

    :::python
    >>> L = [11]
    >>> L.append(open('script2.py')) # list.append does not iterate
    >>> L
    [11, <io.TextIOWrapper name='script2.py' mode='r' encoding='cp1252'>]


The **sum** call computes the sum of all the numbers in any iterable; the **any** and **all** built-ins return True if any or all items in an iterable are True, respectively; and **max** and **min** return the largest and smallest item in an iterable, respectively.

    :::python
    >>> sum([3, 2, 4, 1, 5, 0])
    15
    >>> any(['spam', '', 'ni'])
    True
    >>> all(['spam', '', 'ni'])
    False
    >>> max([3, 2, 5, 1, 4])
    5
    >>> min([3, 2, 5, 1, 4])
    1


Argument-unpacking syntax in calls accepts iterables, it’s also possible to use the **zip** built-in to *unzip* zipped tuples, by making prior or nested **zip** results arguments for another zip call.

    :::python
    >>> X = (1, 2)
    >>> Y = (3, 4)
    >>>
    >>> list(zip(X, Y))  # Zip tuples: returns an iterable
    [(1, 3), (2, 4)]
    >>>
    >>> A, B = zip(*zip(X, Y))  # Unzip a zip!
    >>> A
    (1, 2)
    >>> B
    (3, 4)


##New Iterables in Python 3.X
One of the fundamental distinctions of Python 3.X is its stronger emphasis on iterators than 2.X.

Specifically, in addition to the iterators associated with built-in types such as files and dictionaries, the dictionary methods **keys, values**, and **items** return iterable objects in Python 3.X, as do the built-in functions **range, map, zip**, and **filter**. As shown in the prior section, the last three of these functions both return iterables and process them. All of these tools produce results on demand in Python 3.X, instead of constructing result lists as they do in 2.X.
