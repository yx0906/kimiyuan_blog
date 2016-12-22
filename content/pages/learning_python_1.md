Title: Getting Started
Date: 2016-09-01 23:00
Modified: 2016-09-01 23:00
Category: Python
Tags: readingnotes, learningpython
Slug: learning-python-1
Authors: Kimi Yuan
Summary: Why Do People Use Python? How to Run Python?

[TOC]

This is a reading note for CHAPTER 1-3 in the book **Learning Python, 5th Edition** by *Mark Lutz*.

# A Python Q&A Session

## Why Do People Use Python?

* Software quality
* Developer productivity
* Program portability
* Support libraries
* Component integration
* Enjoyment

## What Can I Do with Python?

* Systems Programming
* GUIs
* Internet Scripting
* Component Integration
* Database Programming
* Rapid Prototyping
* Numeric and Scientific Programming
* And More: Gaming, Images, Data Mining, Robots, Excel...


---
# How Python Runs Programs


##Python's traditional runtime execution model

source code you type is translated to byte code, which is then run by the Python Virtual Machine. Your code is automatically compiled, but then it is interpreted.

Python saves byte code (“.pyc” means compiled “.py” source) like this as a startup speed optimization. The next time you run your program, Python will load the .pyc files and skip the compilation step, as long as you haven't changed your source code since the byte code was last saved, and aren't running with a different Python than the one that created the byte code. It works like this:

![Python runtime model]({filename}/images/Python_runtime_model.jpg)

* ***Source changes***: Python automatically checks the last-modified timestamps of source
* ***Python versions***


---
# How You Run Programs

## Unix Script Basics

Unix-style executable scripts are just normal text files containing Python statements, but with two special properties:

* **Their first line is special.** Scripts usually start with a line that begins with the characters **#!** (often called “hash bang” or “shebang”), followed by the path to the Python interpreter on your machine.
* **They usually have executable privileges.** Script files are usually marked as executable to tell the operating system that they may be run as top-level programs. On Unix systems, a command such as chmod *+x file.py* usually does the trick.

Let's look at an example for Unix-like systems. Use your text editor again to create a file of Python code called *brian*:

	:::python
	#!/usr/local/bin/python
	print('The Bright Side ' + 'of Life...')

If you give the file executable privileges with a **chmod +x brian** shell command, you can run it from the operating system shell as though it were a binary program.

	% brian
	The Bright Side of Life...

##The Unix env Lookup Trick

On some Unix systems, you can avoid hardcoding the path to the Python interpreter in your script file by writing the special first-line comment like this:

	:::python
	#!/usr/bin/env python
	...script goes here...


When coded this way, the *env* program locates the Python interpreter according to your system search path settings (in most Unix shells, by looking in all the directories listed in your **PATH** environment variable). This scheme can be more portable, as you don't need to hardcode a Python install path in the first line of all your scripts. That way, if your scripts ever move to a new machine, or your Python ever moves to a new location, you must update just **PATH**, not all your scripts.

> In Python 2.X, raw_input()

> In Python 3.X, input()

##Module Imports and Reloads

After the first import, later imports do nothing, even if you change and save the module's source file again in another window:

	:::python
	>>> import script1
	win32
	65536
	Spam
	>>> import script1
	>>> import script1


If you really want to force Python to run the file again in the same session without stopping and restarting the session, you need to instead call the **reload** function avail- able in the imp standard library module (this function is also a simple built-in in Python 2.X, but not in 3.X):

	:::python
	>>> from imp import reload # Must load from module in 3.X (only)
	>>> reload(script1)
	win32
	65536
	Spam
	<module 'script1' from '.\\script1.py'>
	>>>


##Debugging Python Code

* **Do nothing.** Read the error message, and go fix the tagged line and file.
* **Insert print statements. **
* **Use IDE GUI debuggers. **
* **Use the pdb command-line debugger.**  In pdb, you type commands to step line by line, display variables, set and clear breakpoints, continue to a breakpoint or error, and so on. You can launch pdb interactively by importing it, or as a top-level script. Either way, because you can type commands to control the session, it provides a powerful debugging tool. pdb also includes a postmortem function (*pdb.pm()*) that you can run after an exception occurs, to get information from the time of the error.
* **Use Python's -i command-line argument.**
