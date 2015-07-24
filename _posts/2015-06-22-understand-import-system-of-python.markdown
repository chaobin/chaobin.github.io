---
layout: post
type: post
title: Understand Import System of Python
tags: Python, Import, Hacking, Import Hook, Python 3, PEP-302
description: In some rare situations, you need to work with Python's import system for some customized behaviours. There are several ways of doing it. The specifications of how the import system works varied from time to time, and more so between Python 2 and Python 3. Here I give you a detailed breakdown, yet hopefully easy to understand.
---

A piece of computer code is first edited, then packaged, distributed, finally imported and executed. Viewing this as a series of stages where you gain different aspects (or level) of control, in Python, the stage of **import** is where you can decide what and how the code is imported into Python interpreter.

Examples what this can be used for:

- on-demand determination on what's importable
- importing code that's not in sys.path yet (because you work in an unsual environment like I do)
- importing code from places other than file system at all

There are several ways of doing it. The specifications of how the import system works varied from time to time, and more so between Python 2 and Python 3. Here I give you a detailed breakdown, yet hopefully easy to understand.

### Invoking import system

Here are three ways a Python programer can invoke the import system in Python: 

```python

import asyncio # using import statement
asyncio = __import__('asyncio') # using builtin __import__
asyncio = importlib.import_module('asyncio') # using import_module from importlib

```

The three ways above are almost identical.

It is **important** to understand that Python's import system is really a complicated system, and the ways shown above are merely a button of this system, a button you press to ask for a module/package to be imported by the system. Because I said these three ways are almost identical, using the button analogy, they are all pressing the same button. In fact, here is what they do:

- an *import* statement is translated into a *\_\_import\_\_* call (Python will search a function named *\_\_import\_\_* in global namespace, failing that, it will use the builtin *\_\_import\_\_*)
- *importlib.import_module()* calls *\_\_import\_\_* internally.

### What's happening

Upon this:

```python

import module # a module
import pkg.module # a module from a package 

```

#### In Python **2.7**, the [import system](https://www.python.org/dev/peps/pep-0302/) does these in this precise order:

1. Try to find *sys.modules["module"]*, return the value if found and the process stops.
1. Look at *sys.meta_path*, which is a list of objects called finders that implement [import protocol](https://www.python.org/dev/peps/pep-0302/#id27), call *find_module("module")* on every object. Right now, this list is empty by default, therefore, this steps does nothing.
1. Call a builtin importer that knows how to import builtin module (Modules that are loaded as part of the Python itself, *sys* for example.)
1. Call a frozen importer that knows how to import [frozen module](https://docs.python.org/2/library/imp.html#imp.init_frozen) (Frozen modules are modules written in Python whose compiled byte-code object is incorporated into a custom-built Python interpreter by Python’s freeze utility).
1. Look at *sys.path_hooks*, which is another list of finders that implement the import protocol, call *find_module("module")* on every object. Right now, this list contains the *ZipImporter* by default that knows how to import module in a zip file.
1. Search places in *sys.path* (you can see there are many things happening before it).
1. Load the module using the loaders (In 2.7, the loader is directly returned) returned by the finders above, and place the module into *sys.modules* for cache.
1. Set a set of module related information on the module,. Things like *\_\_doc\_\_*, *\_\_file\_\_* are set in here.
1. Bind the module object to the name "module", and put it into the caller's current scope.
1. If it is a package, as in the second line *import pkg.module*, the *pkg* is first imported using the steps above, and *"\_\_path\_\_"* is set on it, and later passed into *find_module("pkg.module", path=pkg.\_\_path\_\_)* to import *pkg.module*, this is called a *submodule* import. Submodules are imported this way recursively (As a result of importing pkg.module, the sys.modules will contain both *pkg* and *pkg.module*).

#### In Python **3.4**, the import system does these in this precise order:

1. Try to find *sys.modules["module"]*, return the value if found and the process stops.
1. Look at sys.meta_path, right now in **3.4**, this by default contains:

    1. A *BuiltinImporter* that knows how to import builtin module (Modules that are loaded as part of the Python itself, *sys* for example).
    1. A *FrozenImporter* that knows how to import [frozen module](https://docs.python.org/2/library/imp.html#imp.init_frozen).
    1. A *PathFinder* knows how to find module in import path (*sys.path*, for example)

1. Look at *sys.path_hooks*, which is another list of finders that implement the import protocol, call *find_spec("module")* on every object. Right now, this list contains the *ZipImporter* and *FileFinder*.
1. The rest is roughly the same as in **2.7**, except the *sys.meta_path* is looked up above.

As you can see, the most important difference between **2.7** and **3.4** is the organization of *sys.meta_path*. In **3.4**, The builtin importer and frozen importer, along with the path finder is all organized into *sys.meta_path*, it looks simpler and more organized from outside.

### Thread safety

*sys.modules* and *sys.path* will be manipulated during the invocation of import system. Because it is shared by all threads, synchronization is applied internally. Module *imp* has several function to work with the lock and states.

### Interfaces of Import System

Python exposes its import system to the developer, with even simpler interface in latest 3.x versions. Here are modules you can use:

- [*imp*](https://docs.python.org/2/library/imp.html), the builtin module that has the lowest level of semantics (or import system machinaries, called by Python). Things like aquiring the import lock, releasing it, creating a new module object, load code from source file, bytes-compiled file, load dynamic library (SOs on Unix, DLLs on Windows), etc. It's used to closely access the internals of import system.
- [*importlib*](http://docs.python.jp/3/library/importlib.html), the standard module provides higher level semantics working with the import system. An example would be to call *importlib.import_module()* instead of calling *\_\_import\_\_*  traditionally. In other words, you should always first consider *importlib* if you, for some reason, don't want to directly use the *import* statement.


### An example using this understanding

The following is an import hook that, on-demand, looks for the latest version of the module/pkg on the file system, and add the path to it into sys.path, then import it. It uses the understanding above on import system, places the hook properly so that it doesn't get in the way of import system trying to import builtin module, standard library, or those already available in sys.path.

```python

import sys
import os
import imp
import importlib

PY27 = (sys.version_info.major == 2 and sys.version_info.minor == 7)
PY34 = (sys.version_info.major == 3 and sys.version_info.minor == 4)
assert PY27 or PY34

ROOT = '/Users/cbt/Projects/importers_in_python/test'

class OnDemandPathFinder(object):

    def __init__(self, path):
        if path != ROOT:
            raise ImportError(
                "OnDemandPathFinder only works for %s" % ROOT)

    def find_module(self, fullname, path=None):
        '''
        Manipulates sys.path on demand.
        '''
        print('OnDemandPathFinder tries to find %s' % fullname)
        # 1. find the path of the latest version
        top_level_name = fullname.split('.', 1)[0]
        path_pkg = os.path.join(ROOT, top_level_name)
        if not os.path.exists(path_pkg):
            return None
        latest_release = max(os.listdir(path_pkg))
        path_latest_release = os.path.join(path_pkg, latest_release)
        # 2. add that path into sys.path
        sys.path.append(path_latest_release)
        if PY34:
            # 3. retry sys.meta_path
            for importer in sys.meta_path:
                if isinstance(importer, self.__class__):
                    continue
                loader = importer.find_module(fullname, path)
                if not loader is None:
                    return loader
        elif PY27:
            # 3. Using imp to load module for me
            return self

    def load_module(self, name):
        print('loading %s' % name)
        return imp.load_module(name, *imp.find_module(name))

if PY34:
    sys.meta_path.append(OnDemandPathFinder(ROOT))
elif PY27:
    # Because of the differences between 2.7 and 3.4,
    # register into sys.path_hooks so that the builtin
    # and stdlibs do not invoke this finder,
    # which would slow down import system.
    sys.path_hooks.append(OnDemandPathFinder)
    sys.path.append(ROOT)


```

Here we can test it:

```python

def test():
    import sys
    import importlib
    import smtplib
    import pkg1.module
    print(pkg1.module.__file__)
    import pkg2.module
    print(pkg2.module.__file__)
    import pkg3

test()
```

My 3.4.3 and 2.7.10 gave me this:

```

OnDemandPathFinder tries to find pkg1
loading pkg1
/Users/cbt/Projects/importers_in_python/test/pkg1/2.0/pkg1/module.pyc
OnDemandPathFinder tries to find pkg2
loading pkg2
/Users/cbt/Projects/importers_in_python/test/pkg2/2.0/pkg2/module.pyc
OnDemandPathFinder tries to find pkg3
Traceback (most recent call last):
  File "my_importer.py", line 74, in <module>
    test()
  File "my_importer.py", line 72, in test
    import pkg3
ImportError: No module named pkg3

```

### References:

- [https://www.python.org/dev/peps/pep-0302/](https://www.python.org/dev/peps/pep-0302/)
- [https://docs.python.org/3/reference/import.html](https://docs.python.org/3/reference/import.html)
- [https://docs.python.org/3/library/importlib.html](https://docs.python.org/3/library/importlib.html)

