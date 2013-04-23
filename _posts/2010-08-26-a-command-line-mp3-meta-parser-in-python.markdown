---
layout: post
type: post
title: A command line MP3 meta info parser written in Python
tags: python, mp3, meta info, parser, linux, command line
description: Using hsautiotag, which is a meta info parser written in pure Python supporting mp3/mp4/wmv/flac/ogg, here is a command line mp3 parser I wrote. You can actually try this out by executing it.
---

Using [hsautiotag](http://pypi.python.org/pypi/hsaudiotag3k/1.0.1 "hsautiotag"), which is a meta info parser written in pure Python supporting mp3/mp4/wmv/flac/ogg, here is a command line mp3 parser I wrote. You can actually try this out by executing it.

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: cbtchn@gmail.com

from hsaudiotag import mpeg
from google.apputils import app 
from fabulous.color import *
import os

G = green
R = red 
NULL = 'Unknown'

def main(argv):
    if len(argv) < 2:
        raise SystemExit(R('USAGE: songtag [file1] [file2] ...'))
    files = argv[1:]
    if files[0] == '*':
        files = [f for f in os.listdir('.') if f.endswith('.mp3')]
    info = meta(files)
    display(info)

def meta(mp3_list):
    info = {}
    for mp3 in mp3_list:
        try:
            song = mpeg.Mpeg(open(mp3))
            info[mp3] = {}
            info[mp3]['Album'] = song.tag.album or NULL
            info[mp3]['Artist'] = song.tag.artist or NULL
            info[mp3]['Year'] = song.tag.year or NULL
            info[mp3]['Genre'] = song.tag.genre or NULL
        except Exception, ex: 
            if info.has_key(mp3): del info[mp3]
            continue
    return info

def display(info):
    for mp3, meta in info.items():
        print(G('-+'*17)) # I like number 7
        print(mp3)
        for tag, value in meta.items():
            print '  %s:  %s' % (tag, value)

if __name__ == '__main__':
    app.run()
```