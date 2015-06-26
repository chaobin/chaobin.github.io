---
layout: post
type: post
title: distutils.version.LooseVersion unorderable in Python 3
tags: Python 3, distutils, language, TypeError
description: Python3 stopped supporting comparison between values of different types, because the implicit type conversion in Python2 that made it work is bad. This language uniform however breaks things, such as distutils.version.LooseVersion.
---

Python3 stopped supporting comparison between values of different types, so:

```python

'a' > 1

```

will now raise *TypeError*.

The idea behind this evolution is that the implicit type conversion in Python2 that made it work is bad. **Javascript** programer knows the difference between these two forms of value comparisons:

```javascript

1 == "1" // true

1 === "1" // false

```

This language uniform however breaks things, such as *distutils.version.LooseVersion*.

```python

from distutils.version import LooseVersion

# TypeError, although this version naming scheme isn't quite usual.
LooseVersion("1.2a") > LooseVersion("1.2.1")

```

The [discussion](https://bugs.python.org/issue14894) in here will deliver a patch into distutils that addresses this so it is backward-compatible with Python 27 where this does not fail, essentially adding code to support comparison between mismatched types.

Before that patch, here is what might realisticly
solve this problem:


```python

from distutils.version import LooseVersion

class LooserVersion(LooseVersion):

    #override
    def _cmp(self, other):
        try:
            return super(LooserVersion, self)._cmp(other)
        except TypeError:
            # When unorderable, compare the version strings instead
            if isinstance(other, str):
                other = LooseVersion(other)
            if self.vstring == other.vstring:
                return 0
            if self.vstring < other.vstring:
                return -1
            if self.vstring > other.vstring:
                return 1

def test():
    versions = ['1.1.1', '1.1bc']
    versions = [LooserVersion(v) for v in versions]
    print(max(versions))

if __name__ == '__main__':
    test()

```
