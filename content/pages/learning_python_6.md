Title: Function Basics and Scopes
Category: Python
Tags: readingnotes, learningpython
Slug: learning-python-6
Authors: Kimi Yuan
Summary: Introduce core ideas behind function definition, the behavior of function call expressions, and the notion and benefits of polymorphism in Python functions. And study key concept related to functions: scopes.

[TOC]

This is a reading note for CHAPTER 16 & 17 in the book **Learning Python, 5th Edition** by *Mark Lutz*.

##Why Use Functions?
* **Maximizing code reuse and minimizing redundancy**
Because they allow us to code an operation in a single place and use it in many places, Python functions are the most basic factoring tool in the language: they allow us to reduce code redundancy in our programs, and thereby reduce maintenance effort.

* **Procedural decomposition**
In general, functions are about procedure—how to do something, rather than what you’re doing it to.

Polymorphism, a term we first met in [Introducing Python Object Types]( http://kimiyuan.com/learning-python-2.html#polymorphism) that essentially means that the meaning of an operation depends on the objects being operated upon.

We code to object interfaces in Python, not data types.
This polymorphic behavior has in recent years come to also be known as duck typing—the essential idea being that your code is not supposed to care if an object is a duck, only that it quacks. Anything that quacks will do, duck or not, and the implementation of quacks is up to the object.

## Advantages of function

* Putting the code in a function makes it a tool that you can run as many times as you like.
Because callers can pass in arbitrary arguments, functions are general enough to work on any two sequences (or other iterables) you wish to intersect.

* When the logic is packaged in a function, you have to change code in only one place if you ever need to change the way the intersection works.

* Coding the function in a module file means it can be imported and reused by any program run on your machine.

---
## Name Resolution: The LEGB Rule

* Name assignments create or change local names by default.
* Name references search at most four scopes: **local**, then **enclosing** functions (if any), then **global**, then **built-in**.
* Names declared in global and nonlocal statements map assigned names to enclosing module and function scopes, respectively.

![Scope LEGB]({filename}/images/scope.jpg)
*The LEGB scope lookup rule. When a variable is referenced, Python searches for it in this order: in the local scope, in any enclosing functions’ local scopes, in the global scope, and finally in the built-in scope. The first occurrence wins. The place in your code where a variable is assigned usually determines its scope. In Python 3.X, nonlocal declarations can also force names to be mapped to enclosing function scopes, whether assigned or not.*

Python’s name-resolution scheme is sometimes called the **LEGB rule**, after the scope names:

* When you use an unqualified name inside a function, Python searches up to four scopes—the local (L) scope, then the local scopes of any enclosing (E) defs and lambdas, then the global (G) scope, and then the built-in (B) scope—and stops at the first place the name is found. If the name is not found during this search, Python reports an error.

* When you assign a name in a function (instead of just referring to it in an expression), Python always creates or changes the name in the local scope, unless it’s declared to be global or nonlocal in that function.

* When you assign a name outside any function (i.e., at the top level of a module file, or at the interactive prompt), the local scope is the same as the global scope—the module’s namespace.

### Nested Scope Details
* **A reference** (X) looks for the name X first in the current local scope (function); then in the local scopes of any lexically enclosing functions in your source code, from inner to outer; then in the current global scope (the module file); and finally in the built-in scope (the module builtins). global declarations make the search begin in the global (module file) scope instead.

* **An assignment** (X = value) creates or changes the name X in the current local scope, by default. If X is declared global within the function, the assignment creates or changes the name X in the enclosing module’s scope instead. If, on the other hand, X is declared nonlocal within the function in 3.X (only), the assignment changes the name X in the closest enclosing function’s local scope.

```
    X = 99                   # Global scope name: not used
    def f1():
        X = 88               # Enclosing def local
        def f2():
           print(X)         # Reference made in nested def
        f2()
```

## Program Design
### Minimize Global Variables
In general, functions should rely on arguments and return values instead of globals.

Global variables are probably the most straightforward way in Python to retain shared state information—information that a function needs to remember for use the next time it is called.

Programs that use multithreading to do parallel processing in Python also commonly depend on global variables—they become shared memory between functions running in parallel threads, and so act as a communication device.

### Minimize Cross-File Changes
The best way to communicate across file boundaries is to call functions, passing in arguments and getting back return values. In this specific case, we would probably be better off coding an accessor function to manage the change:

    :::python
    # first.py
    X = 99

    def setX(new):            # Accessor make external changes explit
        global X              # And can manage access in a single place
        X = new

    # second.py
    import first
    first.setX(88)            # Call the function instead of changing directly

## Factory Functions: Closures
In this code, the call to action is really running the function we named f2 when f1 ran. This works because functions are objects in Python like everything else, and can be passed back as return values from other functions. Most importantly, f2 remembers the enclosing scope’s X in f1, even though f1 is no longer active.

The function object in question remembers values in enclosing scopes regardless of whether those scopes are still present in memory. In effect, they have attached packets of memory (a.k.a. *state retention*), which are local to each copy of the nested function created, and often provide a simple alternative to classes in this role.

    :::python
    def f1():
        X = 88
        def f2():
            print(X)         # Remembers X in enclosing def scope
        return f2            # Return f2 but don't call it

    action = f1()            # Make, return function
    action()                 # Call it now: prints 88

    >>> def maker(N):
            def action(X):                    # Make and return action
                return X ** N                 # action retains N from enclosing scope
            return action

    >>> f = maker(2)                          # Pass 2 to argument N
    >>> f
    <function maker.<locals>.action at 0x0000000002A4A158>

    >>> f(3)                                  # Pass 3 to X, N remembers 2: 3 ** 2
    9
    >>> f(4)                                  # 4 ** 2
    16

    >>> g = maker(3)                          # g remembers 3, f remembers 2
    >>> g(4)                                  # 4 ** 3
    64
    >>> f(4)                                  # 4 ** 2
    16


### Loop variables may require defaults, not scopes

If a **lambda** or **def** defined within a function is nested inside a loop, and the nested function references an enclosing scope variable that is changed by that loop, all functions generated within the loop will have the same value—the value the referenced variable had in the **last** loop iteration. In such cases, you must still use **defaults** to save the variable’s **current** value instead.

    :::python
    >>> def makeActions():
            acts = []
            for i in range(5):                       # Tries to remember each i
                acts.append(lambda x: i ** x)        # But all remember same last i!
            return acts

    >>> acts = makeActions()
    >>> acts[0]
    <function makeActions.<locals>.<lambda> at 0x0000000002A4A400>

    >>> acts[0](2)                                   # All are 4 ** 2, 4=value of last i
    16
    >>> acts[1](2)                                   # This should be 1 ** 2 (1)
    16
    >>> acts[2](2)                                   # This should be 2 ** 2 (4)
    16
    >>> acts[4](2)                                   # Only this should be 4 ** 2 (16)
    16

This is one case where we still have to explicitly retain enclosing scope values with default arguments, rather than enclosing scope references. We must pass in the current value of the enclosing scope's variable with a default. Because defaults are evaluated when the nested function is **created** (not when it's later *called*), each remembers its own value for **i:**

    :::python
    >>> def makeActions():
            acts = []
            for i in range(5):                       # Use defaults instead
                acts.append(lambda x, i=i: i ** x)   # Remember current i
            return acts

    >>> acts = makeActions()
    >>> acts[0](2)                                   # 0 ** 2
    0
    >>> acts[1](2)                                   # 1 ** 2
    1
    >>> acts[2](2)                                   # 2 ** 2
    4
    >>> acts[4](2)                                   # 4 ** 2



## The nonlocal Statement in 3.X

The **nonlocal** statement is similar in both form and role to **global,** covered earlier. Like **global, nonlocal** declares that a name will be changed in an enclosing scope. Unlike global, though, nonlocal applies to a name in an enclosing function’s scope, not the global module scope outside all **defs**. Also unlike global, nonlocal names must already exist in the enclosing function’s scope when declared—they can exist only in enclosing functions and cannot be created by a first assignment in a nested **def**.

In other words, **nonlocal** both allows assignment to names in enclosing function scopes and limits scope lookups for such names to enclosing **defs**.

    :::python
    >>> def tester(start):
            state = start             # Each call gets its own state
            def nested(label):
                nonlocal state        # Remembers state in enclosing scope
                print(label, state)
                state += 1            # Allowed to change it if nonlocal
            return nested

    >>> F = tester(0)
    >>> F('spam')                     # Increments state on each call
    spam 0
    >>> F('ham')
    ham 1
    >>> F('eggs')
    eggs 2

## State Retention Options

In summary, **globals, nonlocals, classes**, and **function attributes** all offer changeable state-retention options. Globals support only single-copy shared data; nonlocals can be changed in 3.X only; classes require a basic knowledge of OOP; and both classes and function attributes provide portable solutions that allow state to be accessed directly from outside the stateful callable object itself.

### State with Globals: A Single Copy Only

    :::python
    >>> def tester(start):
            global state                   # Move it out to the module to change it
            state = start                  # global allows changes in module scope
            def nested(label):
                global state
                print(label, state)
                state += 1
            return nested

    >>> F = tester(0)
    >>> F('spam')                          # Each call increments shared global state
    spam 0
    >>> F('eggs')
    eggs 1

This works in this case, but it requires global declarations in both functions and is prone to name collisions in the global scope (what if “state” is already being used?). A worse, and more subtle, problem is that it only allows for a single shared copy of the state information in the module scope—if we call tester again, we’ll wind up resetting the module’s state variable, such that prior calls will see their state overwritten:

    :::python
    >>> G = tester(42)                     # Resets state's single copy in global scope
    >>> G('toast')
    toast 42

    >>> G('bacon')
    bacon 43

    >>> F('ham')                           # But my counter has been overwritten!
    ham 44


### State with nonlocal: 3.X only

    :::python
    >>> def tester(start):
            state = start                  # Each call gets its own state
            def nested(label):
                nonlocal state             # Remembers state in enclosing scope
                print(label, state)
                state += 1                 # Allowed to change it if nonlocal
            return nested

    >>> F = tester(0)
    >>> F('spam')                          # State visible within closure only
    spam 0
    >>> F.state
    AttributeError: 'function' object has no attribute 'state'


### State with Classes: Explicit Attributes

    :::python
    >>> class tester:                          # Class-based alternative (see Part VI)
            def __init__(self, start):         # On object construction,
                self.state = start             # save state explicitly in new object
            def nested(self, label):
                print(label, self.state)       # Reference state explicitly
                self.state += 1                # Changes are always allowed

    >>> F = tester(0)                          # Create instance, invoke __init__
    >>> F.nested('spam')                       # F is passed to self
    spam 0
    >>> F.nested('ham')
    ham 1

    >>> G = tester(42)                         # Each instance gets new copy of state
    >>> G.nested('toast')                      # Changing one does not impact others
    toast 42
    >>> G.nested('bacon')
    bacon 43

    >>> F.nested('eggs')                       # F's state is where it left off
    eggs 2
    >>> F.state                                # State may be accessed outside class
    3

We could also make our class objects look like callable functions using operator overloading ```. __call__ ```intercepts direct calls on an instance, so we don’t need to call a named method:

    :::python
    >>> class tester:
            def __init__(self, start):
                self.state = start
            def __call__(self, label):         # Intercept direct instance calls
                print(label, self.state)       # So .nested() not required
                self.state += 1

    >>> H = tester(99)
    >>> H('juice')                             # Invokes __call__
    juice 99
    >>> H('pancakes')
    pancakes 100



### State with Function Attributes: 3.X and 2.X

    :::python
    >>> def tester(start):
            def nested(label):
                print(label, nested.state)     # nested is in enclosing scope
                nested.state += 1              # Change attr, not nested itself
            nested.state = start               # Initial state after func defined
            return nested

    >>> F = tester(0)
    >>> F('spam')                              # F is a 'nested' with state attached
    spam 0
    >>> F('ham')
    ham 1
    >>> F.state                                # Can access state outside functions too
    2

    >>> G = tester(42)                         # G has own state, doesn't overwrite F's
    >>> G('eggs')
    eggs 42
    >>> F('ham')
    ham 2

    >>> F.state                                # State is accessible and per-call
    3
    >>> G.state
    43
    >>> F is G                                 # Different function objects
    False
