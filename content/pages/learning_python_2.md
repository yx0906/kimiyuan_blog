Title: Introducing Python Object Types
Category: Python
Tags: readingnotes, learningpython
Slug: learning-python-2
Authors: Kimi Yuan
Summary: Brief Introduction about Core Types like Strings, Lists, Dictionaries, etc in Python

[TOC]

This is a reading note for CHAPTER 4 in the book **Learning Python, 5th Edition** by *Mark Lutz*.

## The Python Conceptual Hierarchy
1. Programs are composed of modules.
2. Modules contain statements.
3. Statements contain expressions.
4. Expressions create and process objects.

## Python's Core Data Types

| Object type | Example literals/creation |
| : --- | : --- |
| Numbers | 1234, 3.1415, 3+4j, 0b111, Decimal(), Fraction() |
| Strings | 'spam', "Bob's", b'a\x01c', u'sp\xc4m' |
| Lists | [1, [2, 'three'], 4.5],list(range(10)) |
| Dictionaries | dict(hours=10), `{'food': 'spam', 'taste': 'yum'}`  |
| Tuples | (1, 'spam', 4, 'U'),tuple('spam'),named tuple |
| Files | open('eggs.txt'),open(r'C:\ham.bin', 'wb')  |
| Sets | set('abc'),{'a', 'b', 'c'} |
| Other core types | Booleans, types, None |
| Program unit types | Functions, modules, classes (Part IV, Part V, Part VI) |
| Implementation-related types |  Compiled code, stack tracebacks (Part IV, Part VII) |

### Getting Help
* dir(s)
* help(s.replace)

## String
Strings also support an advanced substitution operation known as **formatting**, available as both an expression (the orig- inal) and a string method call (new as of 2.6 and 3.0); the second of these allows you to omit relative argument value numbers as of 2.7 and 3.1:

    :::python
    >>> '%s, eggs, and %s' % ('spam', 'SPAM!')
    'spam, eggs, and SPAM!'
    >>> '{0}, eggs, and {1}'.format('spam', 'SPAM!')
    'spam, eggs, and SPAM!'
    >>> '{}, eggs, and {}'.format('spam', 'SPAM!')
    'spam, eggs, and SPAM!'

##List
### Comprehensions

    :::python
    >>> M
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> col2 = [row[1] for row in M] >>> col2
    [2, 5, 8]


    >>> [row[1] + 1 for row in M] # Add 1 to each item in column 2
    [3, 6, 9]
    >>> [row[1] for row in M if row[1] % 2 == 0] # Filter out odd items
    [2, 8]


**map** makes a list of results all at once instead, and is not needed in other contexts that iterate automatically, unless multiple scans or list-like behavior is also required:

    :::python
    >>> list(map(sum, M)) # Map sum over items in M
    [6, 15, 24]

In Python 2.7 and 3.X, comprehension syntax can also be used to create sets and dictionaries:

    :::python
    >>> {sum(row) for row in M} # Create a set of row sums
    {24, 6, 15}
    >>> {i : sum(M[i]) for i in range(3)} # Creates key/value table of row sums
    {0: 6, 1: 15, 2: 24}

In fact, lists, sets, dictionaries, and generators can all be built with comprehensions in 3.X and 2.7:

    :::python
    >>> [ord(x) for x in 'spaam']
    [115, 112, 97, 97, 109]
    >>> {ord(x) for x in 'spaam'}
    {112, 97, 115, 109}

    >>> {x: ord(x) for x in 'spaam'}
    {'p': 112, 'a': 97, 's': 115, 'm': 109}

    >>> G = (ord(x) for x in 'spaam')
    >>> G
    <generator object <genexpr> at 0x103da8500>
    >>> next(G)
    115
    >>> next(G)
    112
    >>> next(G)
    97
    >>> next(G)
    97
    >>> next(G)
    109
    >>> next(G)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    StopIteration


## Dictionaries
### Missing Keys: if Tests

    :::python
    >>> if not 'f' in D: print('missing')  ## D is a dictionary missing

### Sorting Keys: for Loops

The built-in function **sorted** call returns the result and sorts a variety of object types, in this case sorting dictionary keys automatically:

    :::python
    >>> D
    {'a': 1, 'c': 3, 'b': 2}
    >>> for key in sorted(D): print(key, '=>', D[key])
    a => 1
    b => 2
    c => 3


### Iteration and Optimization
In a nutshell, an object is *iterable* if it is either a physically stored sequence in memory, or an object that generates one item at a time in the context of an iteration operation —a sort of “virtual” sequence. More formally, both types of objects are considered iterable because they support the iteration protocol—they respond to the **iter** call with an object that advances in response to **next** calls and raises an exception when finished producing values.

The *generator* comprehension expression we saw earlier is such an object: its values aren't stored in memory all at once, but are produced as requested, usually by iteration tools. Python file objects similarly iterate line by line when used by an iteration tool: file content isn't in a list, it's fetched on demand. Both are iterable objects in Python— a category that expands in 3.X to include core tools like **range** and **map**.

A major rule of thumb in Python is to code for **simplicity** and **readability** first and worry about performance later, after your program is working, and after you've proved that there is a genuine performance concern. More often than not, your code will be quick enough as it is. If you do need to tweak code for performance, though, Python includes tools to help you out, including the **time** and **timeit** modules for timing the speed of alternatives, and the profile module for isolating bottlenecks.

## Files

    :::python
    >>> for line in open('script2.py'):
    ... print(line.upper(), end='')


Notice that the **print** uses end=**''** here to suppress adding a \n, because line strings already have one (without this, our output would be double-spaced; in 2.X, a trailing comma (**,**)works the same as the **end**). This is considered the best way to read text files line by line today, for three reasons: it's the simplest to code, might be the quickest to run, and is the best in terms of memory usage.

## Other Core Types
### Sets

Sets are neither mappings nor sequences; rather, they are unordered collections of unique and immutable objects.

    :::python
    >>> X = set('spam') # Make a set out of a sequence in 2.X and 3.X
    >>> Y = {'h', 'a', 'm'}   # Make a set with set literals in 3.X and 2.7

    >>> X, Y     # A tuple of two sets without parentheses
    ({'m', 'a', 'p', 's'}, {'m', 'a', 'h'})

    >>> X & Y    # Intersection
    {'m', 'a'}
    >>> X | Y    # Union
    {'m', 'h', 'a', 'p', 's'}
    >>> X - Y    # Difference
    {'p', 's'}
    >>> X > Y    # Superset
    False

### Decimal and fraction
Python recently grew a few new numeric types: decimal numbers, which are fixed-precision floating-point numbers, and fraction numbers, which are rational numbers with both a numerator and a denominator.

    :::python
    >>> 1 / 3           # Floating-point (add a .0 in Python 2.X)
    0.3333333333333333
    >>> (2/3) + (1/2)
    1.1666666666666665

    >>> import decimal    # Decimals: fixed precision
    >>> d = decimal.Decimal('3.141')
    >>> d + 1
    Decimal('4.141')

    >>> decimal.getcontext().prec = 2
    >>> decimal.Decimal('1.00') / decimal.Decimal('3.00')
    Decimal('0.33')

    >>> from fractions import Fraction   # Fractions: numerator+denominator
    >>> f = Fraction(2, 3)
    >>> f + 1
    Fraction(5, 3)
    >>> f + Fraction(1, 2)
    Fraction(7, 6)

## Polymorphism

in Python, we code to object *interfaces* (operations supported), not to types. That is, we care what an object *does*, not what it is. Not caring about specific types means that code is automatically applicable to many of them—any object with a compatible interface will work, regardless of its specific type.
