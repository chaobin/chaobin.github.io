---
layout: post
type: post
title: A parser implemented in a state machine pattern to parse "fenced" text block.
tags: python, text, parser, pattern, state machine
description: State machine is one of the simplest yet most useful patterns when solving some complicated problems. Such as this one, where I implemented a text parser that parses the "fenced" text block out of one piece of text. This "fenced" text block thing is often used to embed some special text block right into one larger text so that this "fenced" text block can be differently further processed. (Think about the code block in markdown.) Look at the code and see how simple it is to implement this state machine, and it is also very extensible and versatile. 
---

State machine is one of the simplest yet most useful patterns when solving some complicated problems. Such as this one, where I implemented a text parser that parses the "fenced" text block out of one piece of text. This "fenced" text block thing is often used to embed some special text block right into one larger text so that this "fenced" text block can be differently further processed. (Think about the code block in markdown.) Look at the code and see how simple it is to implement this state machine, and it is also very extensible and versatile. However, IMPORTANTLY, in this specific implementation of mine using Python(CPython) programming language, the Python interpreter creates a stack whenever a function is called (On contrary, Stackless Python makes effort to eliminate this), so there is a limited number of states can be "accessed". In some other languages, such as [Lua](http://www.lua.org/pil/6.3.html "Proper Tail Calls"), or [Erlang](http://www.erlang.org/doc/efficiency_guide/myths.html#id61519 "Tail-recursive functions are MUCH faster than recursive functions"), comes with an implementation called "tail call", that is a very special RETURN statement which is not at a cost of creating a stack for the returned function call.

```python
# -*- coding: utf-8 -*-
# author: cbtchn@gmail.com

'''
Parser that parses code-snippets from text.
'''

import re
START_CODE = re.compile('<source-(?P<lang>\S+)>')
END_CODE = re.compile('<\/source-(?P<lang>\S+)>')


class Parser(object):
    '''
    A state machine that processes code nodes from texts.
    '''

    def __init__(self, text=None):
        self.text = text
        self.start_offset = 0
        self.end_offset = 0
        self.start_node = START_CODE
        self.end_node = END_CODE
        self.last_end = 0 
        self.drawer = []

    def in_node(self):
        ''' 
        State when encounters a starting node.
        '''
        self.start_offset = self.start_node.search(self.text, self.start_offset)
        if self.start_offset is None:
            return self.done()
        return self.out_node()

    def out_node(self):
        ''' 
        State when encounters an ending node.
        '''
        self.end_offset = self.end_node.search(self.text, self.start_offset.start())
        if self.end_offset is None:
            return self.done()
        return self.transition()

    def transition(self):
        ''' 
        Put the code into the drawer.
        '''
        self.drawer.append(self.text[self.last_end:self.start_offset.start()])
        code = self.text[self.start_offset.end(): self.end_offset.start()]
        self.drawer.append((self.start_offset.groups()[0], code))
        self.start_offset = self.end_offset.end()
        self.last_end = self.end_offset.end()
        return self.in_node()

    def done(self):
        '''
        State when the state flow comes to the end.
        '''
        self.drawer.append(self.text[self.last_end:])
        return self.drawer

    def parse(self):
        '''
        Console of the state flow.
        '''
        return self.in_node()
```

Here is an example of using this parser:

```python
>>> source = ''' <source-python> import os </source-python> ''' 
>>> p = Parser(source)
>>> p.parse()
['', ('python', 'import os'), '']
```