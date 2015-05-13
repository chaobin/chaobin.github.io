---
layout: post
type: post
title: Inter-operate with C/C++ in Python
tags: Python, C/C++, wrapping
description: It's in Python's early days that its ability to inter-operate with lower level languages made it promising. Later, many solutions emerged to improve this ability in some way.
---

This article outlines several existing options, providing a 
high level comparison.

It's in Python's early days that its ability to inter-operate with
lower level languages made it promising. Later, many solutions emerged
to improve this ability in some way.

Out of many, we have these that are more relevant these days:

- Standard Python C APIs
- boost.python
- CFFI
- Cython
- SWIG

### Python C APIs

Exposing your APIs to Python by wrapping them. The wrapping is
a routine of calling several reoccuring Python C APIs that does
this in sequence:

- Unpacking the function paramters from tuple and dictionary
- Calling your C/C++ APIs with the arguments
- Converting the return value of your C/C++ calls into Python
values
- Free any resources
- Return the result

After wrapping all the desired APIs, you need to write a
boiler-plate module for Python to be able to pick up all the
bindings.

#### Pros

- Technically straightforward
- Well documented
- No gotchas

#### Conts

- Lots of C code
- Dealing directly with Python C APIs requires one to
know about it fairly well, such as how it manages resources,
Python 2 and Python 3 incompatibilities

### boost.python

[Boost.Python](http://www.boost.org/doc/libs/1_58_0/libs/python/doc/index.html)
is made specifically for inter-operating with C++ code in Python. It is a 
C++ template library that provides a declarative syntax to wrap up your code,
then cleverly exposing these to Python using Python C API. Its cleverness
includes *type conversion*, *exception translation*, and more.
Using it is straightforward. You write a reasonably thin layer of ad-hoc code
in your C++ code to define the APIs that you want to export as Python APIs.

An example:

```C++

class_<World>("World")
    .def("greet", &World::greet)
    .def("set", &World::set)

```

[Here](https://github.com/TNG/boost-python-examples) has a number of 
demos that can give you a better look.

#### Pros

- The biggest of that would be it is made specifically to work with C++.
- No other language to learn. You only have to know your own C++ code
and read boost.python docs, then you are ready to go.
- It does certain things for you, such as type conversion, exception 
translation, abstraction of Python C API (so that you don't have to
deal with them directly)

#### Cons

- It is not one of the most performant solutions as far as that's concerned
- Using the similar approach, SWIG can actually further reduce
the code one has to write in C++ for your C++ library to have
a Python interface.

### CFFI

[CFFI](https://cffi.readthedocs.org/en/latest/) is one of the modern solutions
that combines several nice things. It tremendously reduces what one needs to
do to give the C/C++ library a Python interface, it is fast, and it frees
you from having to know about Python C APIs and its internals.

Its *cffi.cdef()*, *cffi.dlopen()*, *cffi.verify()* are almost all that you
need to be working with, and it is declarative.

#### ABI compatibility with cffi.dlload

```python

>>> from cffi import FFI
>>> ffi = FFI()
>>> ffi.cdef("""
...     int printf(const char *format, ...); // copy-pasted from the man page
... """)
>>> C = ffi.dlopen(None)               # loads the entire C namespace
>>> arg = ffi.new("char[]", "world")   # equivalent to C code: char arg[] = "world";
>>> C.printf("hi there, %s!\n", arg)   # call printf
hi there, world!

```

The *cffi.dlopen(SHARED_LIB)* can be used to load shared library (.so)
and from there, generate the Python interface that you can access
using the dynamic library returned by *cffi.dlopen()*. CFFI doesn't recommend
users to use this approach as this requires ABI compatibilities which
are less persistent than API compatibilities.

#### API compatibility with cffi.verify

```python

from cffi import FFI
ffi = FFI()
ffi.cdef("""     // some declarations from the man page
    struct passwd {
        char *pw_name;
        ...;
    };
    struct passwd *getpwuid(int uid);
""")
C = ffi.verify("""   // passed to the real C compiler
#include <sys/types.h>
#include <pwd.h>
""", libraries=[])   # or a list of libraries to link with
p = C.getpwuid(0)
assert ffi.string(p.pw_name) == 'root'    # on Python 3: b'root'

```

The *cffi.verify()* is the heart of CFFI. CFFI calls the C compiler
**just-in-time** to compile the source and return a dynamic library.
The cffi.verify() takes everything that revolves around using
GCC to build a library, -I, -L, and other flags.

Starting from **0.9**, *cffi.verify()* supports C++ code. Because
the current latest version is **0.9.2**, C++ support hasn't gone
through a time long enough to receive many real world usage though.

Underlying, CFFI relies on the same libffi that Python's ctypes does.
But it aproaches differently to be more reliable and performant.
[Here](http://eli.thegreenplace.net/2013/03/09/python-ffi-with-ctypes-and-cffi)
is an article that explains nicely how CFFI works.

[Here](https://bitbucket.org/cffi/cffi/src/default/demo) has number of
demos that should quickly give you a better look.

#### Pros

- It is built on an existing approach that's successful in Lua
- It starts to prove its own success by seeing a lot more libs using it. Many
libraries in PyPI are currently using CFFI.
- It really simplifies the process. You can go from finishing reading its doc
in an hour then being able to interface your reasonably large library into
Python in hours.
- It also abstracts away the Python internals so you don't deal with
Python 2/3 incompatibilities, Python's GC, etc.

#### Cons

- The major concern for now is, CFFI only started to support C++ since 0.9,
and its current latest version is 0.9.2. So its ability with C++ is much less
tested than with C.

### Cython

[Cython](http://docs.cython.org/) is a standalone language.
It is in fact a super set of Python, extending it with:

- C data types (with auto conversion from/to Python values )
- ability to call C/C++ libraries freely
- a compiler that compiles the Cython code into C

Compared to the alternatives listed above, Cython is unique in that it is a
compiler in fact. It first generates the Cython code into a C file, then compiles
that C file into a library. This approach takes a price of being very complex,
but with the advantage of being the most performant solution. In fact, Cython is
most popular in the scientific area of Python, where every library that does
number crunching or matrix/vector manipulation hungers for speed.

Cython is also unique in that it allows one to surgically optimize Python code
by redefining any piece of Python code in Cython, then gain a C-like speed, and
the redefinition is often not much different than the Python version, except
you add type annotations to the function, parameters, and variables, and that
is often trivial to do.

Since **0.13**, it supports C++. The latest version is 0.22.

#### Pros

- It's fast
- It's a versatile tool
- Its syntax is a hybrid of mostly Python, then with C data types, easy for
a experienced Python user to lear
- Very active upstream support and real world usage in large projects, so you
get a lot of support

#### Cons

- It is a new language to learn, and quite a learning curve at first.
- Among its various skills, the ability to wrap a C/C++ library isn't its
selling point. It is fully capable, but entails you to do quite some
coding in its syntax. When it comes to be giving your existing C/C++
library a Python interface, Cython is more of a can-do-that than
does-just-that.

### SWIG

[Short](http://www.swig.org/) for Simplified Wrapper and Interface Generator.
I have heard of SWIG a long time ago, but never used it. At the moment, not
much research has been done on it. But it does look a much more complicated
CFFI in a sense (it shines in that it can generate from an interface file 
the wrapper for quite a few languages). I shall take a look at this some other
day.


