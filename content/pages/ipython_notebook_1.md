Title: Intro to IPython in Jupyter
Category: Python
Tags: IPython, Jupyter, Python
Slug: ipython-1
Authors: Kimi Yuan
Summary: Overview of how to start with IPython in Jupyter and useful tips

[TOC]

# Overview

The goal of IPython is to create a comprehensive environment for **interactive and exploratory computing**. It's now a project under Jupyter's umbrella  To support this goal, IPython has three main components:

* An enhanced interactive Python shell.
* A decoupled two-process communication model, which allows for multiple clients to connect to a computation kernel, most notably the web-based notebook provided with **Jupyter**.
* An architecture for interactive parallel computing now part of the **ipyparallel** package.

The rest of article will focus on main features for interactive part.

# Getting Started

## Installation

Recommended way is to install with Jupyter and then you will get Jupyter Notebook as well.

First, ensure that you have the latest pip; older versions may have trouble with some dependencies:

```
$ pip3 install --upgrade pip
```
Then install the Jupyter Notebook using:
```
$ pip3 install jupyter
```
(Use pip if using legacy Python 2.)

Congratulations. You have installed Jupyter Notebook.

## Launch

```
$ jupyter console
```
or
```
$ ipython
```

# Interactive IPython

## Tab completion

Tab completion, especially for attributes, is a convenient way to explore the structure of any object you’re dealing with. Simply type `object_name.<TAB>` to view the object’s attributes. Besides Python objects and keywords, tab completion also works on file and directory names.

## Exploring your objects

Typing `object_name?` will print all sorts of details about any object, including docstrings, function definition lines (for call arguments) and constructor details for classes. Using `??` provides additional detail.

    :::python
    In [222]: a?
    Type:        list
    String form: [3, 4, 21, 1]
    Length:      4
    Docstring:
    list() -> new empty list
    list(iterable) -> new list initialized from iterable's items
    
    In [223]: min?
    Signature: min(*args, **kwargs)
    Docstring: <no docstring>
    File:      /<ipython-input-122-3906a7bf4db8>
    Type:      function
    
    In [224]: min??
    Signature: min(*args, **kwargs)
    Source:
    def min(*args, **kwargs):
        key = kwargs.get("key", lambda x:x)
        if len(args) == 1:
            items = args[0]
        else:
            items = args
        result = next(iter(items))
        for item in items:
            if key(item) < key(result):
                result = item
        return result
    File:      /<ipython-input-122-3906a7bf4db8>
    Type:      function

## Magic functions
IPython has a set of predefined *‘magic functions’* that you can call with a command line style syntax. There are two kinds of magics, line-oriented and cell-oriented. **Line magics** are prefixed with the **%** character and work much like OS command-line calls: they get as an argument the rest of the line, where arguments are passed without parentheses or quotes. **Lines magics** can return results and can be used in the right hand side of an assignment. **Cell magics** are prefixed with a double **%%**, and they are functions that get as an argument not only the rest of the line, but also the lines below it in a separate argument.

Magics are useful as convenient functions where Python syntax is not the most natural one, or when one want to embed invalid python syntax in their work flow.

The following examples show how to call the builtin **%timeit** magic, both in line and cell mode:

Line magic:

    :::python
    In [1]: %timeit range(1000)
    1000000 loops, best of 3: 468 ns per loop

It equals to the following but much simpler.

    $ python3 -m timeit -n 1000000 "range(1000)"
    1000000 loops, best of 3: 0.361 usec per loop

Cell magic:

    :::python
    In [2]: %%timeit x = range(10000)
       ...: max(x)
       ...:
    1000 loops, best of 3: 223 us per loop
    The builtin magics include:


For more details on any magic function, call **%somemagic?** to read its docstring. To see all the available magic functions, call **%lsmagic**.

### %recall

Repeat a command, or get command to input line for editing.

    %recall 45

Place history line 45 on the next input prompt. Use %hist to find out the number.

    %recall 1-4

Combine the specified lines into one cell, and place it on the next input prompt. See %history for the slice syntax.

    %recall foo+bar

If foo+bar can be evaluated in the user namespace, the result is placed at the next input prompt. Otherwise, the history is searched for lines which contain that substring, and the most recent one is placed at the next input prompt.

### %rerun

Re-run previous input.

By default, you can specify ranges of input history to be repeated (as with %history). With no arguments, it will repeat the last line.

## Running

The **%run** magic command allows you to run any python script and load all of its data directly into the interactive namespace. Since the file is re-read from disk each time, changes you make to it are reflected immediately (unlike imported modules, which have to be specifically reloaded). IPython also includes dreload, a recursive reload function, like `from IPython.lib.deepreload import reload as dreload`


`%run` has special flags for timing the execution of your scripts (-t), or for running them under the control of either Python’s pdb debugger (-d) or profiler (-p).

## History

### Input and Output
Input and output history are kept in variables called In and Out, keyed by the prompt numbers.

* `_i`, `_ii`, `_iii`: store previous, next previous and next-next previous inputs. The last three objects in output history are also kept in variables named `_`, `__` and `___`.

```
In [14]: 1**2
Out[14]: 1

In [15]: 2**2
Out[15]: 4

In [16]: 3**2
Out[16]: 9

In [17]: _i, _ii , _iii, _, __, ___,
Out[17]: ('3**2', '2**2', '1**2', 9, 4, 1)
```

* `_ih`, `In`,: a list of all inputs; `_ih[n]` or `In[n]` is the input from line *n*. The results of output are also stored in a global dictionary (not a list, since it only has entries for lines which returned a result) available under the names `_oh` and `Out` (similar to `_ih` and `In`)

* Additionally, global variables named `_i<n>` are dynamically created (<n> being the prompt counter). Global variables named `_<n>` are dynamically created (<n> being the prompt counter), such that the result of output <n> is always available as `_<n>` (don’t use the angle brackets, just the number, e.g. `_21`).

* As an input, `_i<n> == _ih[<n>] == In[<n>]` and as an output, `_<n> == _oh[<n>] == Out[<n>]`.


```
In [6]: foo = "Hello World!"

In [7]: foo
Out[7]: 'Hello World!'

In [8]: _i7, _ih[7], In[7], _7, _oh[7], Out[7]
Out[8]: ('foo', 'foo', 'foo', 'Hello World!', 'Hello World!', 'Hello World!')
```

* If you overwrite In with a variable of your own, you can remake the assignment to the internal list with a simple `In=_ih`. Out variable you can recover it by typing `Out=_oh` at the prompt.

### %history

A history function **%history** allows you to see any part of your input history by printing a range of the `_i` variables.

You can also search (‘grep’) through your history by typing `%hist -u -g somestring`. `-u` show only unique history. This is handy for searching for URLs, IP addresses, etc. You can bring history entries listed by ‘%hist -g’ up for editing with the **%recall** command, or run them immediately with **%rerun**.


## System shell commands

To run any command at the system shell, simply prefix it with `!`, e.g.:

```
!ping www.bbc.co.uk
```

You can assign the result of a system command to a Python variable with the syntax `myfiles = !ls`. Similarly, the result of a magic (as long as it returns a value) can be assigned to a variable. To explicitly get this sort of output without assigning to a variable, use two exclamation marks (`!!ls`)

IPython also allows you to expand the value of python variables when making system calls. Wrap variables or expressions in **{** braces **}**:


    In [1]: pyvar = 'Hello world'
    In [2]: !echo "A python variable: {pyvar}"
    A python variable: Hello world
    In [3]: import math
    In [4]: x = 8
    In [5]: !echo {math.factorial(x)}
    40320

To explicitly get this sort of output without assigning to a variable, use two exclamation marks (`!!ls`)
However, `!!` commands cannot be assigned to a variable.


    In [1]: myfiles = !ls
    In [2]: myfiles
    Out[2]:
    ['Applications',
     'Desktop',
     'Documents',
     'Downloads',
     'Library',
     'Movies',
     'Music',
     'Pictures',
     'Public'
