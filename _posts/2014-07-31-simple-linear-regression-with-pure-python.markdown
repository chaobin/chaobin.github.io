---
layout: post
type: post
title: Simple Linear Regression with Pure Python
tags: algorithm, machine learning, prediction, python
description: A pure Python implementation of the simple linear regression with one variable. A short and dependency-free Python recipe that offers the ability to predict using simple linear regression.
---

Linear regression is a simple yet very useful tool for value prediction.
The outcome of the regression is a best fitting line that minimizes the squared errors.
The popular Stanford Machine Learning course offered on Coursera is an
accesible, in-depth, and systematic material to start learning machine learning,
and it uses the Gradient Descent to gradually approach to the best fitting line.
Instead, there is a statistical approach that directly solves the best fitting line function.

```python

def fit3(X, Y):

    def mean(Xs):
        return sum(Xs) / len(Xs)
    m_X = mean(X)
    m_Y = mean(Y)

    def std(Xs, m):
        normalizer = len(Xs) - 1
        return math.sqrt(sum((pow(x - m, 2) for x in Xs)) / normalizer)
    # assert np.round(Series(X).std(), 6) == np.round(std(X, m_X), 6)

    def pearson_r(Xs, Ys):

        sum_xy = 0
        sum_sq_v_x = 0
        sum_sq_v_y = 0

        for (x, y) in zip(Xs, Ys):
            var_x = x - m_X
            var_y = y - m_Y
            sum_xy += var_x * var_y
            sum_sq_v_x += pow(var_x, 2)
            sum_sq_v_y += pow(var_y, 2)
        return sum_xy / math.sqrt(sum_sq_v_x * sum_sq_v_y)
    # assert np.round(Series(X).corr(Series(Y)), 6) == np.round(pearson_r(X, Y), 6)

    r = pearson_r(X, Y)

    b = r * (std(Y, m_Y) / std(X, m_X))
    A = m_Y - b * m_X

    def line(x):
        return b * x + A
    return line

```

[Here](http://nbviewer.ipython.org/url/128.199.200.181/download/Linear%20Regression%20-%20simple%20regression.ipynb) is the online notebook that shows the plot.
