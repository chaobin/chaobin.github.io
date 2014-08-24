---
layout: post
type: post
title: Printing spiral numbers with Javascript
tags: algorithm, javascript, number, game
description: Printing a pattern with spiral numbers starting from the center. If you have seen one snail, or a picture of it, you know what this is all about. Also, the algorithm used in this implementation most importantly sees the correlation between the square root (namely the pattern size) and the coordinates of the point. This has an advantage where the implementation might look straightforward, while the disadvantage is that it calculates based on the position, not the numbers. A better solution would be to start with the number, say 1, 2, 3, 4, ... Anyway, see how it plays out in this implementation.
---

Printing a pattern with spiral numbers starting from the center. If you have seen one snail, or a picture of it, you know what this is all about. Also, the algorithm used in this implementation most importantly sees the *correlation between the square root (namely the pattern size) and the coordinates of the point*. This has an advantage where the implementation might look straightforward, while the disadvantage is that it calculates based on the position, not the numbers. A better solution would be to start with the number, say 1, 2, 3, 4, ... Anyway, see how it plays out in this implementation.

Here lies the [code](https://github.com/chaobin/ulamespiral "Ulame Spiral").

Usage -

```javascript

var sn = new snail.Snail(9);
sn.prepare();
sn.draw();
snail.ui.draw(sn._zeros);
snail.ui.animate(sn);
```

Here is an animated demonstrate -

<div>
  <style type="text/css" media="screen">
    .point {
      width: 30px;
      float: left;
    }
  </style>
  <script type="text/javascript" src="/javascripts/snail.js"></script>
  <script type="text/javascript" charset="utf-8">
    var sn = new snail.Snail(9);
    sn.prepare();
    sn.draw();
    snail.ui.draw(sn._zeros);
    snail.ui.animate(sn);
  </script>
</div>
