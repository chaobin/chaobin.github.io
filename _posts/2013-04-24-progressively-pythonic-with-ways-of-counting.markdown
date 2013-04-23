---
layout: post
type: post
title: Progressively pythonic with ways of counting elements
tags: python, pythonic, programing, learning, frequency distribution
description: Frequency Distribution is very useful in some occasions. e.g., server log analysis, natural language processing. Here are several approaches in Python to calculate the frequency distribution, demonstrating a progressively pythonic programming style.
---

Frequency Distribution is very useful in some occasions. e.g., server log analysis, natural language processing. Here are several approaches in Python to calculate the frequency distribution, demonstrating a progressively pythonic programming style.

Now we have our data and we need to count the occurrences of each word it contains - 

```python
words = "apple banana apple strawberry banana lemon"
```

```python
# fundamental approach - 
list_words = words.split()
unique_words = set(list_words)
freqs = dict(
  (w, list_words.count(w))
  for w in unique_words
)
```

Or, if you have python version >= *2.7*, you have a new powerful syntax called *set comprehension*, which you use it like *list comprehension*, but with *{* ... *}* and it evaluates to a set instance -

```python
list_words = words.split()
# note the curly parenthesis
freqs = { w:list_words.count(w) for w in list_words }
```

The above code of counting all relied on *list.count()* method, which could be of high time complexity because it counts the word on every occurrence. Here is an improvement on that -

```python
from collections import defaultdict


freqs = defaultdict(int)
for w in words.split():
    freqs[w] += 1
```

Again, if you have python version >= *2.7*, you have a *Counter* class from *collections* module that does exactly this -

```python
from collections import Counter


freqs = Counter(words.split()) # It has a set of dict-like API
```

The example code above should have shown several different ways of counting the occurrences of elements using different data structure and their APIs. However, it really is a more complex answer to the question that which one is **better**, though the last one using the *Counter* may exude better readability.
The complexity to answer this question lies at the performance of execution. I will try to talk more about this in a **separate** post.