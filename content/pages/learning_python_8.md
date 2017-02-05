Title: Advanced Function Topics
Category: Python
Tags: readingnotes, learningpython
Slug: learning-python-8
Authors: Kimi Yuan
Summary:
Status: draft

[TOC]

## Recursive Functions

    :::python
    >>> def mysum(L):
            if not L:
                return 0
            else:
                return L[0] + mysum(L[1:])           # Call myself recursively

    >>> mysum([1, 2, 3, 4, 5])
    15

    def mysum(L):
        return 0 if not L else L[0] + mysum(L[1:])           # Use ternary expression

    def mysum(L):
        return L[0] if len(L) == 1 else L[0] + mysum(L[1:])  # Any type, assume one

    def mysum(L):
        first, *rest = L
        return first if not rest else first + mysum(rest)    # Use 3.X ext seq assign


The latter two of these fail for empty lists but allow for sequences of any object type that supports +, not just numbers:

    :::python
    >>> mysum([1])                              # mysum([]) fails in last 2
    1
    >>> mysum([1, 2, 3, 4, 5])
    15
    >>> mysum(('s', 'p', 'a', 'm'))             # But various types now work
    'spam'
    >>> mysum(['spam', 'ham', 'eggs'])
    'spamhameggs'


Standard Python limits the depth of its runtime call stack—crucial to recursive call programs—to trap infinite recursion errors. To expand it, use the sys module:

    :::python
    >>> sys.getrecursionlimit()         # 1000 calls deep default
    1000
    >>> sys.setrecursionlimit(10000)    # Allow deeper nesting
    >>> help(sys.setrecursionlimit)     # Read more about it



## Generator



* Generator functions

A function **def** statement that contains a **yield** statement is turned into a generator function. When called, it “returns a new *generator object* with automatic retention of local scope and code position; an automatically created `__iter__` method that simply returns itself; and an automatically created `__next__` method (next in 2.X) that starts the function or resumes it where it last left off, and raises **StopIteration** when finished producing results.

* Generator expressions

A comprehension expression enclosed in parentheses is known as a generator expression. When run, it returns a new generator object with the same automatically created method interface and state retention as a generator function call’s results—with an `__iter__` method that simply returns itself; and a `__next__` method (next in 2.X) that starts the implied loop or resumes it where it last left off, and raises **StopIteration** when finished producing results.

### Generators Are Single-Iteration Objects

Both generator functions and generator expressions are their own iterators and thus support just *one active iteration*—unlike some built-in types, you can’t have multiple iterators of either positioned at different locations in the set of results. Because of this, a generator’s iterator is the generator itself.
