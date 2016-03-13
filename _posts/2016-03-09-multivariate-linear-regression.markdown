---
layout: post_mathjax
type: post
title: Continued example on Linear Regression with multiple variable
tags: machine learning, gradient descent, linear regression
description: Using the Gradient Descent, here is the continued example from the previous post but with a linear model with multiple variable.
---

Using the Gradient Descent, here is the continued example from the previous post but with a linear model with multiple variable.

The code used in this post can be found in [here](https://github.com/chaobin/isaac.git).


### Feature selection

The many different models will all face one problem, that is to decide which feature to use. In the context of machine learning, this topic is called **feature selection**, or rather, **feature engineering** in large. It is often said that selecting the proper set of features is more important than fitting the parameters, as by training the model, we are only approaching the optimum of precision that is already determined by the feature set we introduced into the model. Therefore, it is rather important to investigate the feature set used in a model before the significant work is carried on training.

There are many techniques and algorithms we can use in feature selection. The following will, omitting further discussing on the topic, use one of the algorithms called Pearson Correlation to order the features and use the top ranking ones in our multivariable regression example.

```python
# Here I used pandas just to simplify the process
# of retrieving and preprocess data. I will then
# get the internal numpy narray to work with thereafter.
import math
import pandas as pd
from pandas import DataFrame, Series

URL_GPA = "http://onlinestatbook.com/2/case_studies/data/sat.txt"

def online_gpa():
    df = DataFrame.from_csv(URL_GPA, sep=' ', header=0, index_col=None)
    return df

df_gpa = online_gpa()
column_names = list(df_gpa.columns)
data = df_gpa.values
```

```python
# pandas has a corr() that by default uses the pearson method
# to calculate the correlation pair-wise.
# Here we do that and take the two most correlated
# columns out of the data and use them in our
# multivariate model.

features = df_gpa.corr()['univ_GPA'].sort_values(ascending=False)[1:3].index.tolist()
columns = [column_names.index(f) for f in features]


%pylab inline
from importlib import reload
from isaac.models import linear
from isaac.pipeline import preprocess
from isaac.plots import basic as plots
```

```python

model = linear.Regression.from_dimension(len(features) + 1)
X, Y = preprocess.get_XY_from_frame(data, columns)
train_size = 0.7
train_size = math.ceil(train_size)
_X, _Y = X[:train_size], Y[:train_size]


# the untrained, initialized model
predictions = model.predict(X)
cost = model.costs(X, Y)
title = "cost: %f" % cost
with plots.zoom_plot(10, 6):
    plots.plot_predictions_3d(X, Y, predictions, features, title=title)
    plots.plot_predictions_3d(X, Y, predictions, features, title=title, mirror=True)
```

![png](/images/posts/2_multivariate_linear_regression_5_0.png)

![png](/images/posts/2_multivariate_linear_regression_5_1.png)


```python
from isaac.optimizers import gradient


descent = gradient.Descent(model, _X, _Y, 0.001)


descent.run(10)
predictions = model.predict(X)
cost = model.costs(X, Y)
title = "cost: %f" % cost
with plots.zoom_plot(10, 6):
    plots.plot_predictions_3d(X, Y, predictions, features, title=title)
    plots.plot_predictions_3d(X, Y, predictions, features, title=title, mirror=True)
```

![png](/images/posts/2_multivariate_linear_regression_8_0.png)

![png](/images/posts/2_multivariate_linear_regression_8_1.png)


```python
descent.run(50)
predictions = model.predict(X)
cost = model.costs(X, Y)
title = "cost: %f" % cost
with plots.zoom_plot(10, 6):
    plots.plot_predictions_3d(X, Y, predictions, features, title=title)
    plots.plot_predictions_3d(X, Y, predictions, features, title=title, mirror=True)
```

![png](/images/posts/2_multivariate_linear_regression_9_0.png)

![png](/images/posts/2_multivariate_linear_regression_9_1.png)


```python
# Let's use the bottom ranked features and compare
# the model with one above


features = df_gpa.corr()['univ_GPA'].sort_values(ascending=True)[1:3].index.tolist()
columns = [column_names.index(f) for f in features]


model = linear.Regression.from_dimension(len(features) + 1)
X, Y = preprocess.get_XY_from_frame(data, columns)
train_size = 0.7
train_size = math.ceil(train_size)
_X, _Y = X[:train_size], Y[:train_size]
```

```python
descent = gradient.Descent(model, _X, _Y, 0.001)
```

```python
# untrained model
predictions = model.predict(X)
cost = model.costs(X, Y)
title = "cost: %f" % cost
with plots.zoom_plot(10, 6):
    plots.plot_predictions_3d(X, Y, predictions, features, title=title)
    plots.plot_predictions_3d(X, Y, predictions, features, title=title, mirror=True)
```

![png](/images/posts/2_multivariate_linear_regression_14_0.png)

![png](/images/posts/2_multivariate_linear_regression_14_1.png)


```python
descent.run(10)
predictions = model.predict(X)
cost = model.costs(X, Y)
title = "cost: %f" % cost
with plots.zoom_plot(10, 6):
    plots.plot_predictions_3d(X, Y, predictions, features, title=title)
    plots.plot_predictions_3d(X, Y, predictions, features, title=title, mirror=True)
```

![png](/images/posts/2_multivariate_linear_regression_15_0.png)

![png](/images/posts/2_multivariate_linear_regression_15_1.png)


```python
descent.run(50)
predictions = model.predict(X)
cost = model.costs(X, Y)
title = "cost: %f" % cost
with plots.zoom_plot(10, 6):
    plots.plot_predictions_3d(X, Y, predictions, features, title=title)
    plots.plot_predictions_3d(X, Y, predictions, features, title=title, mirror=True)
```

![png](/images/posts/2_multivariate_linear_regression_16_0.png)

![png](/images/posts/2_multivariate_linear_regression_16_1.png)

