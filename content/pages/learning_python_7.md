Title: Arguments
Date: 2016-09-17 14:00
Modified: 2016-09-17 15:00
Category: Python
Tags: readingnotes, learningpython
Slug: learning-python-7
Authors: Kimi Yuan
Summary: Study the concepts in Python argument passing—the way that objects are sent to functions as inputs. 

[TOC]

This is a reading note for CHAPTER 18 in the book **Learning Python, 5th Edition** by *Mark Lutz*.

## Argument-Passing Basics

* Arguments are passed by automatically assigning objects to local variable names.
* Assigning to argument names inside a function does not affect the caller.
* Changing a mutable object argument in a function may impact the caller.

* Immutable arguments are effectively passed "by value".
* Mutable arguments are effectively passed "by pointer".

### Avoiding Mutable Argument Changes

If we don’t want in-place changes within functions to impact objects we pass to them, though, we can simply make explicit copies of mutable objects.

    :::python
    def changer(a, b):
        b = b[:]            # Copy input list so we don't impact caller
        a = 2
        b[0] = 'spam'       # Changes our list copy only

    L = [1, 2]
    changer(X, tuple(L))    # Pass a tuple, so changes are errors


## Special Argument-Matching Modes

The steps that Python internally carries out to match arguments before assignment can roughly be described as follows:

1. Assign nonkeyword arguments by position.
2. Assign keyword arguments by matching names.
3. Assign extra nonkeyword arguments to ***name** tuple.
4. Assign extra keyword arguments to ****name** dictionary.
5. Assign default values to unassigned arguments in header.

### Keyword and Default Examples

    :::python
    def func(spam, eggs, toast=0, ham=0):   # First 2 required
        print((spam, eggs, toast, ham))

    func(1, 2)                              # Output: (1, 2, 0, 0)
    func(1, ham=1, eggs=0)                  # Output: (1, 0, 0, 1)
    func(spam=1, eggs=0)                    # Output: (1, 0, 0, 0)
    func(toast=1, eggs=2, spam=3)           # Output: (3, 2, 1, 0)
    func(1, 2, 3, 4)                        # Output: (1, 2, 3, 4)

Notice again that when keyword arguments are used in the call, the order in which the arguments are listed doesn’t matter; Python matches by name, not by position.

### Arbitrary Arguments Examples

Headers: Collecting arguments
*args collects unmatched positional arguments into a tuple.
**args only works for keyword arguments and collects them into a dictionary.

Calls: Unpacking arguments
We can use the * syntax when we call a function, too. In this context, its meaning is the inverse of its meaning in the function definition—it unpacks a collection of arguments, rather than building a collection of arguments. For example, we can pass four arguments to a function in a tuple and let Python unpack them into individual arguments:

    :::python
    >>> def func(a, b, c, d): print(a, b, c, d)

    >>> args = (1, 2)
    >>> args += (3, 4)
    >>> func(*args)                            # Same as func(1, 2, 3, 4)
    1 2 3 4

Similarly, the ** syntax in a function call unpacks a dictionary of key/value pairs into separate keyword arguments:

    :::python
    >>> args = {'a': 1, 'b': 2, 'c': 3}
    >>> args['d'] = 4
    >>> func(**args)                           # Same as func(a=1, b=2, c=3, d=4)
    1 2 3 4

Again, we can combine normal, positional, and keyword arguments in the call in very flexible ways:

    :::python
    >>> func(*(1, 2), **{'d': 4, 'c': 3})      # Same as func(1, 2, d=4, c=3)
    1 2 3 4
    >>> func(1, *(2, 3), **{'d': 4})           # Same as func(1, 2, 3, d=4)
    1 2 3 4
    >>> func(1, c=3, *(2,), **{'d': 4})        # Same as func(1, 2, c=3, d=4)
    1 2 3 4
    >>> func(1, *(2, 3), d=4)                  # Same as func(1, 2, 3, d=4)
    1 2 3 4
    >>> func(1, *(2,), c=3, **{'d':4})         # Same as func(1, 2, c=3, d=4)
    1 2 3 4


## Python 3.X Keyword-Only Arguments

Python 3.X generalizes the ordering rules in function headers to allow us to specify *keyword-only arguments*—arguments that must be passed by keyword only and will never be filled in by a positional argument.

Keyword-only arguments must be specified after a single star, not two—named arguments cannot appear after the ****args** arbitrary keywords form in the arguments list.

    :::python
    >>> def f(a, *b, **d, c=6): print(a, b, c, d)          # Keyword-only before **!
    SyntaxError: invalid syntax

    >>> def f(a, *b, c=6, **d): print(a, b, c, d)          # Collect args in header

    >>> f(1, 2, 3, x=4, y=5)                               # Default used
    1 (2, 3) 6 {'y': 5, 'x': 4}

    >>> f(1, 2, 3, x=4, y=5, c=7)                          # Override default
    1 (2, 3) 7 {'y': 5, 'x': 4}

    >>> f(1, 2, 3, c=7, x=4, y=5)                          # Anywhere in keywords
    1 (2, 3) 7 {'y': 5, 'x': 4}

    >>> def f(a, c=6, *b, **d): print(a, b, c, d)          # c is not keyword-only here!

    >>> f(1, 2, 3, x=4)
    1 (3,) 2 {'x': 4}


    >>> def f(a, *b, c=6, **d): print(a, b, c, d)          # KW-only between * and **

    >>> f(1, *(2, 3), **dict(x=4, y=5))                    # Unpack args at call
    1 (2, 3) 6 {'y': 5, 'x': 4}

    >>> f(1, *(2, 3), **dict(x=4, y=5), c=7)               # Keywords before **args!
    SyntaxError: invalid syntax

    >>> f(1, *(2, 3), c=7, **dict(x=4, y=5))               # Override default
    1 (2, 3) 7 {'y': 5, 'x': 4}

    >>> f(1, c=7, *(2, 3), **dict(x=4, y=5))               # After or before *
    1 (2, 3) 7 {'y': 5, 'x': 4}

    >>> f(1, *(2, 3), **dict(x=4, y=5, c=7))               # Keyword-only in **
    1 (2, 3) 7 {'y': 5, 'x': 4}
