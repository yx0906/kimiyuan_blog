Title: Strings
Category: Python
Tags: readingnotes, learningpython, python
Slug: learning-python-4
Authors: Kimi Yuan
Summary: In-depth tour of the string object type. More about coding string literals and string operations.

[TOC]

This is a reading note for CHAPTER 7 in the book **Learning Python, 5th Edition** by *Mark Lutz*.

## String Literals

* Single quotes: 'spa"m'
* Double quotes: "spa'm"
* Triple quotes: '''... spam ...''', """... spam ..."""
* Escape sequences: "s\tp\na\0m"
* Raw strings: r"C:\new\test.spm"
* Bytes literals in 3.X and 2.6+ (see Chapter 4, Chapter 37): b'sp\x01am'
* Unicode literals in 2.X and 3.3+ (see Chapter 4, Chapter 37): u'eggs\u0020spam'

Python automatically concatenates adjacent string literals in any expression, although it is almost as simple to add a + operator between them to invoke concatenation explicitly.

    :::python
    >>> title = "Meaning " 'of' " Life"        # Implicit concatenation
    >>> title
    'Meaning of Life'


**Escape Sequences Represent Special Characters**

| Escape | Meaning |
| :------- | :------------ |
| \newline |  Ignored (continuation line) |
| \\ | Backslash (stores one \)|
| \' | Single quote (stores ') |
| \" | Double quote (stores ") |
| \a  | Bell |
| \b |  Backspace |
| \f | Formfeed |
| \n |Newline (linefeed) |
| \r | Carriage return |
| \t | Horizontal tab |
| \v | Vertical tab |
| \xhh | Character with hex value hh (exactly 2 digits) |
|\ooo | Character with octal value ooo (up to 3 digits) |
| \0 | Null: binary 0 character (doesn’t end string) |
| \N{ id } | Unicode database ID |
| \uhhhh | Unicode character with 16-bit hex value |
| \Uhhhhhhhh | Unicode character with 32-bit hex value[a] |
| \other | Not an escape (keeps both \ and other) |

**Raw Strings Suppress Escapes**

    :::python
    myfile = open(r'C:\new\text.dat', 'w')

**Triple Quotes Code Multiline Block Strings**

    :::python
    >>> menu = """spam     # comments here added to string!
    ... eggs               # ditto
    ... """
    >>> menu
    'spam     # comments here added to string!\neggs               # ditto\n'

    >>> menu = (
    ... "spam\n"           # comments here ignored
    ... "eggs\n"           # but newlines not automatic
    ... )
    >>> menu
    'spam\neggs\n'


* Useful anytime you need **multiline** text in your program.
* Used for **documentation** strings
* **Temporarily disable** lines of code during development

###Strings in Action
####Indexing and Slicing

![slicing.jpg]({filename}/images/slicing.jpg)
*Offsets and slices: positive offsets start from the left end (offset 0 is the first item), and negatives count back from the right end (offset −1 is the last item). Either kind of offset can be used to give positions in indexing and slicing operations.*

Indexing (S[i]) fetches components at offsets:
* The first item is at offset 0.
* Negative indexes mean to count backward from the end or right.
* S[0] fetches the first item.
* S[−2] fetches the second item from the end (like S[len(S)−2]).

Slicing (S[i:j]) extracts contiguous sections of sequences:
* The upper bound is noninclusive.
* Slice boundaries default to 0 and the sequence length, if omitted.
* S[1:3] fetches items at offsets 1 up to but not including 3.
* S[1:] fetches items at offset 1 through the end (the sequence length).
* S[:3] fetches items at offset 0 up to but not including 3.
* S[:−1] fetches items at offset 0 up to but not including the last item.
* S[:] fetches items at offsets 0 through the end—making a top-level copy of S.
Extended slicing (S[i:j:k]) accepts a step (or stride) k, which defaults to +1:
* Allows for skipping items and reversing order.

**Character code conversions**
The **ord** function convert a single character to its underlying integer code (e.g., its ASCII byte value).
The **chr** function performs the inverse operation, taking an integer code and converting it to the corresponding character

    :::python
    >>> ord('s')
    115
    >>> chr(115)
    's'


###String Formatting Expressions
    :::python
    >>> exclamation = 'Ni'
    >>> 'The knights who say %s!' % exclamation         # String substitution
    'The knights who say Ni!'

    >>> '%d %s %g you' % (1, 'spam', 4.0)               # Type-specific substitutions
    '1 spam 4 you'

    >>> '%s -- %s -- %s' % (42, 3.14159, [1, 2, 3])     # All types match a %s target
    '42 -- 3.14159 -- [1, 2, 3]

*Table 7-4. String formatting type codes*

| Code | Meaning |
| :----- | :---------------------- |
| s | String (or any object’sstr(X)string) |
| r | Same ass, but usesrepr, notstr |
| c | Character (int or str) |
| d | Decimal (base-10 integer) |
| i | Integer |
| u | Same as d(obsolete: no longer unsigned) |
| o | Octal integer (base 8) |
| x | Hex integer (base 16) |
| X | Same asx, but with uppercase letters |
| e | Floating point with exponent, lowercase E Same ase, but uses uppercase letters |
| f | Floating-point decimal |
| F | Same as f, but uses uppercase letters |
| g | Floating-point e or f |
| G | Floating-point E or F |
| % | Literal%(coded as%%) |

The general structure of conversion targets looks like this:

>%[(keyname)][flags][width][.precision]typecode

The type code characters in the first column of Table show up at the end of this target string’s format. Between the % and the type code character, you can do any of the following:
* Provide a *key name* for indexing the dictionary used on the right side of the expression
* List *flags* that specify things like left justification (−), numeric sign (+), a blank before positive numbers and a – for negatives (a space), and zero fills (0)
* Give a total minimum field width for the substituted text
* Set the number of digits (precision) to display after a decimal point for floating-point numbers

Both the *width* and *precision parts* can also be coded as a * to specify that they should take their values from the next item in the input values on the expression’s right side

    :::python
    >>> x = 1.23456789
    >>> '%−6.2f | %05.2f | %+06.1f' % (x, x, x)
    '1.23 | 01.23 | +001.2'

    >>> '%(qty)d more %(food)s' % {'qty': 1, 'food': 'spam'}
    '1 more spam'

    >>> X = '{motto}, {0} and {food}'.format(42, motto=3.14, food=[1, 2])
    >>> X
    '3.14, 42 and [1, 2]'


Here’s the formal structure

>{fieldname component !conversionflag :formatspec}


### String Formatting Method Calls
    :::python
    >>> '{0:10} = {1:10}'.format('spam', 123.4567) # In Python 3.3
    'spam       =  123.4567'
    >>> '{0:>10} = {1:<10}'.format('spam', 123.4567) '
    'spam       = 123.4567  '
    >>> '{0.platform:>10} = {1[kind]:<10}'.format(sys, dict(kind='laptop'))
    '     win32 = laptop    '

{0:10} means the first positional argument in a field 10 characters wide, {1:<10} means the second positional argument left-justified in a 10-character-wide field.

{fieldname component !conversionflag :formatspec} In this substitution target syntax:
* fieldname is an optional number or keyword identifying an argument, which may be omitted to use relative argument numbering in 2.7, 3.1, and later.
* component is a string of zero or more “.name” or “[index]” references used to fetch attributes and indexed values of the argument, which may be omitted to use the whole argument value.
* conversionflagstartswitha!ifpresent,whichisfollowedbyr,s,oratocallrepr, str, or ascii built-in functions on the value, respectively.
* formatspec starts with a : if present, which is followed by text that specifies how the value should be presented, including details such as field width, alignment, padding, decimal precision, and so on, and ends with an optional data type code.

    :::python
    >>> '{0:.{1}f}'.format(1 / 3.0, 4) '0.3333'
    >>> '%.*f' % (4, 1 / 3.0)
    '0.3333'

    >>> '{0:d}'.format(999999999999) '999999999999'
    >>> '{0:,d}'.format(999999999999) '999,999,999,999'


>{fieldname component !conversionflag :formatspec} In this substitution target syntax:

* **fieldname** is an optional number or keyword identifying an argument, which may be omitted to use relative argument numbering in 2.7, 3.1, and later.
* **component** is a string of zero or more **“.name”** or “*[index]*” references used to fetch attributes and indexed values of the argument, which may be omitted to use the whole argument value.
* **conversionflag** starts with a ! if present,which is followed by r,s,or atocallrepr, str, or ascii built-in functions on the value, respectively.
* **formatspec** starts with a : if present, which is followed by text that specifies how the value should be presented, including details such as field width, alignment, padding, decimal precision, and so on, and ends with an optional data type code.
* The formatspec component after the colon character has a rich format all its own, and is formally described as follows (brackets denote optional components and are not coded literally):

>[[fill]align][sign][#][0][width][,][.precision][typecode]

In this, fill can be any fill character other than { or }; align may be <, >, =, or ^, for left alignment, right alignment, padding after a sign character, or centered alignment, respectively; sign may be +, −, or space; and the , (comma) option requests a comma for a thousands separator as of Python 2.7 and 3.1. width and precision are much as in the % expression, and the formatspec may also contain nested {} format strings with field names only, to take values from the arguments list dynamically (much like the * in formatting expressions).
