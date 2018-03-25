Title: The Dynamic Typing Interlude
Category: Python
Tags: readingnotes, learningpython, python
Slug: learning-python-3
Authors: Kimi Yuan
Summary: A deeper look at Python’s dynamic typing model—that is, the way that Python keeps track of object types for us automatically.

[TOC]

This is a reading note for CHAPTER 6 in the book **Learning Python, 5th Edition** by *Mark Lutz*.

At the most practical level, dynamic typing means there is less code for you to write. Just as importantly, though, dynamic typing is also the root of Python’s polymorphism.

##The Case of the Missing Declaration Statements

###Variables, Objects, and References

In sum, variables are created when assigned, can reference any type of object, and must be assigned before they are referenced. This means that you never need to declare names used by your script, but you must initialize names before you can update them; counters, for example, must be initialized to zero before you can add to them.

![assign.jpg]({filename}/images/assign.jpg)
*Names and objects after running the assignment a = 3. Variable a becomes a reference to the object 3. Internally, the variable is really a pointer to the object’s memory space created by running the literal expression 3*

These links from variables to objects are called references in Python—that is, a reference is a kind of association, implemented as a pointer in memory.1 Whenever the variables are later used (i.e., referenced), Python automatically follows the variable-to-object links. This is all simpler than the terminology may imply. In concrete terms:
* Variables are entries in a system table, with spaces for links to objects.
* Objects are pieces of allocated memory, with enough space to represent the values for which they stand.
* References are automatically followed pointers from variables to objects.

### Types Live with Objects, Not Variables

    :::python
    >>> a = 3             # It's an integer
    >>> a = 'spam'        # Now it's a string
    >>> a = 1.23          # Now it's a floating point


*Names* have no types; as stated earlier, types live with objects, not names.

Each *object* also has two standard header fields: a *type designator* used to mark the type of the object, and a *reference counter* used to determine when it’s OK to reclaim the object.

### Objects Are Garbage-Collected

    :::python
    >>>a = 3
    >>>a = 'spam'

In Python, whenever a name is assigned to a new object, the space held by the prior object is reclaimed if it is not referenced by any other name or object. This automatic reclamation of objects’ space is known as garbage collection, and makes life much simpler for programmers of languages like Python that support it.

Internally, Python accomplishes this feat by keeping a counter in every object that keeps track of the number of references currently pointing to that object. As soon as (and exactly when) this counter drops to zero, the object’s memory space is automatically reclaimed.

## Shared References

![shared_ref1]({filename}/images/shared_ref1.jpg)
*Names and objects after next running the assignment b = a. Variable b becomes a reference to the object 3. Internally, the variable is really a pointer to the object’s memory space created by running the literal expression 3.*

![shared_ref2]({filename}/images/shared_ref2.jpg)
*Names and objects after finally running the assignment a = ‘spam’. Variable a references the new object (i.e., piece of memory) created by running the literal expression ‘spam’, but variable b still refers to the original object 3. Because this assignment is not an in-place change to the object 3, it changes only variable a, not b.*

In Python variables are always pointers to objects, not labels of changeable memory areas: setting a variable to a new value does not alter the original object, but rather causes the variable to reference an entirely different object. The net effect is that assignment to a variable itself can impact only the single variable being assigned.

###Shared References and In-Place Changes

    :::python
    >>> L1 = [2, 3, 4]        # A mutable object
    >>> L2 = L1               # Make a reference to the same object
    >>> L1[0] = 24            # An in-place change

    >>> L1                    # L1 is different
    [24, 3, 4]
    >>> L2                    # But so is L2!
    [24, 3, 4]

This sort of change overwrites part of the list object’s value in place. Because the list object is shared by (referenced from) other variables, though, an in-place change like this **doesn’t affect only** L1—that is, you must be aware that when you make such changes, they can impact other parts of your program.

Here are a variety of ways to copy a list, including using the built-in **list** function and the standard library **copy** module. Perhaps the most common way is to **slice** from start to finish.

    :::python
    >>> L1 = [2, 3, 4]
    >>> L2 = L1[:]            # Make a copy of L1 (or list(L1), copy.copy(L1), etc.)
    >>> L1[0] = 24

    >>> L1
    [24, 3, 4]
    >>> L2                    # L2 is not changed
    [2, 3, 4]


To copy a dictionary or set, instead use their **X.copy()** method call, or pass the original object to their type names, **dict** and **set**.

Also, note that the standard library **copy** module has a call for copying any object type generically, as well as a call for copying nested object structures—a dictionary with nested lists, for example:
    :::python
    import copy
    X = copy.copy(Y) # Make top-level "shallow" copy of any object Y
    X = copy.deepcopy(Y) # Make deep copy of any object Y: copy all nested parts


###Shared References and Equality

The **==** operator tests whether the two referenced objects have the same **values**; this is the method almost always used for equality checks in Python. The second method, the **is** operator, instead tests for object **identity**—it returns True only if both names point to the exact same object (compares the pointers that implement references, and it serves as a way to detect shared references), so it is a much stronger form of equality testing and is rarely applied in most programs.

    :::python
    >>> L = [1, 2, 3]
    >>> M = L                 # M and L reference the same object
    >>> L == M                # Same values
    True
    >>> L is M                # Same objects
    True

    >>> L = [1, 2, 3]
    >>> M = [1, 2, 3]         # M and L reference different objects
    >>> L == M                # Same values
    True
    >>> L is M                # Different objects
    False

Python caches and reuses small integers and small strings, as mentioned earlier, the object 42 here is probably not literally reclaimed; instead, it will likely remain in a system table to be reused the next time you generate a 42 in your code.


    :::python
    >>> X = 42
    >>> Y = 42                # Should be two different objects
    >>> X == Y
    True
    >>> X is Y                # Same object anyhow: caching at work!
    True
