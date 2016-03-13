---
layout: post_mathjax
type: post
title: build a logistic model for income prediction
tags: machine learning, gradient descent, logistic regression, cross entropy error
description: Logistic regression is a very popular classification algorithm used in machine learning. A choice of logistic function is used to estimate the probability of the class. Using the same understanding on linear regression and gradient descent previously discussed, it is not too much work to extend that knowledge to understand logistic regression. And in this article, I am building a logistic model to predict the income using an online archive of income data.
---

Logistic regression is a very popular classification algorithm used in machine learning. A choice of *logistic function* is used to estimate the probability of the class. Using the same understanding on linear regression and gradient descent [previously](/2016/03/08/gradient-descent-intuitively-understood/) discussed, it is not too much work to extend that knowledge to understand logistic regression. And in this article, I am building a logistic model to predict the income using an online archive of income data.

### Sigmoid function

The sigmoid function is defined as:

{% raw %}
<div class="equation" data="f(x) = \frac{1}{1 + e^{-x}}"></div>
{% endraw %}


The **sigmoid** function is mathematical tool that squashes a number into a new one within range *(0, 1)*. The logistic regression uses this tool on top of **linear regression** so that the new model will output a value between 0 and 1. And this value is interpreted as the probability of the outcome that is **true** (g(*) > 0.5 or **false** (g(*) < 0.5). Hence logistic regression is used as a **classification algorithm**.

[Previously](http://chaobin.github.io/2016/03/08/gradient-descent-intuitively-understood/), our linear regression model is defined as:

{% raw %}
<div class="equation" data="h_{(\Theta)}(X) = \Theta^T * X"></div>
{% endraw %}

The logistic model can be derived using the sigmoid function:

{% raw %}
<div class="equation" data="h_{(\Theta)}(X) = \frac{1}{1 + e^{-(\Theta^T*X)}}"></div>
{% endraw %}


```python

%pylab inline   
import math
import numpy as np

from isaac.plots import basic as basic_plots

Populating the interactive namespace from numpy and matplotlib

def sigmoid(arr):
    return 1 / (1 + np.exp(- arr))

X = np.arange(-10, 10, 0.01)
Y = sigmoid(X)
basic_plots.plot(X, Y, label="sigmoid", title="The sigmoid function",
                loc="upper left", show=False)
basic_plots.plot(0, 0.5, style='bo')
```

![png](/images/posts/4_logistic_regression_a_realistic_example_4_0.png)


### Interpreting the result

The sigmoid function squashes a value like the graph above similar to what a [heavyside step function](https://en.wikipedia.org/wiki/Heaviside_step_function) does except it is smooth, or continuous, where we take Y = 1 if the value *H > 0.5* and Y = 0 if *H < 0.5*. Because the sigmoid function satisfies several properties, we can also view the result as a logistic distribution and say that:

{% raw %}
<div class="equation" data="P(Y=1 | X) = h_{(\Theta)}(X)"></div>
{% endraw %}

Thus

{% raw %}
<div class="equation" data="P(Y=0 | X) = 1 - P(Y=1 | X)"></div>
{% endraw %}

Thus

{% raw %}
<div class="equation" data="P(Y=0 | X) = 1 - h_{(\Theta)}(X)"></div>
{% endraw %}


That means if we have our hypothesis equals to *0.4*, the interpretation says there is a *40%* chance that Y == 1, or there is *(1 - 40%)* = *60%* chance that Y == 0.

## Cost

Similar to the [Mean Squared Error](http://chaobin.github.io/2016/03/08/gradient-descent-intuitively-understood/) used in describing the error of a linear regression model, we need a meansure of error for the logistic model as well. Intuitively, the error measure:

{% raw %}
<div class="equation" data="J(\Theta)"></div>
{% endraw %}

- should give a relatively larger value to describe a bad prediction
- a smaller value for a good prediction

We are using a measure from information theory called the [cross entropy](We are using a measure from information theory called the [cross entropy](https://en.wikipedia.org/wiki/Cross_entropy():

{% raw %}
<div class="equation" data="J(\Theta) = - log(h_\Theta(x)), y = 1"></div>
{% endraw %}

{% raw %}
<div class="equation" data="J(\Theta) = - log(1 - h_\Theta(x)), y = 0"></div>
{% endraw %}

Combining the two together we get:

{% raw %}
<div class="equation" data="J(\Theta) = -Y * log(h_\Theta(x)) - (1 - Y) * log(1 - h_\Theta(x))"></div>
{% endraw %}

If the cost is then defined over a training data set, we take the *mean* of the summed error:

{% raw %}
<div class="equation" data="J(\Theta) = -\frac{1}{N} \sum_{n=1}^i{ y^{(i)} * log(h_\Theta(x^{(i)})) + (1 - y^{(i)}) * log(1 - h_\Theta(x^{(i)})) }"></div>
{% endraw %}

The following code plots the sigmoid function side by side with the error measure:

```python
Y_1 = Y[Y >= 0.5] # the right half
cost_Y_1 = (- np.log(Y_1))
Y_2 = Y[Y <= 0.5] # the left half
cost_Y_2 = (- (1 - np.log(Y_2)))

basic_plots.plot(X, Y, label="sigmoid", title="The sigmoid function",
                loc="upper left")
plots = (
    (basic_plots.plot, (X[X >= 0], Y_1), {'title': 'h(Theta) >= 0.5, Y = 1'}),
    (basic_plots.plot, (Y_1, cost_Y_1), {
            'title': '- log(h(Theta))',
            'label_xy': ('h(Theta)', 'cost(h(Theta))')}),
    (basic_plots.plot, (X[X <= 0], Y_2), {'title': 'h(Theta) <= 0.5, Y = 0'}),
    (basic_plots.plot, (Y_2, cost_Y_2), {
            'title': '- log(1 - h(Theta))',
            'label_xy': ('h(Theta)', 'cost(h(Theta))')})
)
basic_plots.subplots(2, 2, sharex=False, sharey=False, order='v', plots=plots)
```

![png](/images/posts/4_logistic_regression_a_realistic_example_7_0.png)

![png](/images/posts/4_logistic_regression_a_realistic_example_7_1.png)

Intuitively as the graph above shows, the error measure **decreases** as *h(Theta)* approaches to the value indicating the **correct** output.

The choosing of these error measure also has desired mathematical property, whereas *-log()* is continuous or differentiable and it is a convex function, meaning it has a global minimum.

### Income Prediction

An [archive](http://archive.ics.uci.edu/ml/machine-learning-databases/adult/old.adult.names) of the survey on income was used here. In this post, a logistic model is built, using a subset of features selected by visual guide. The model was then trained using the entire set of the [training data](http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data) contains *32561* entries. The model reports a *16273/16281* accuracy on the [test data](http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test) that contains *16281* entries.

```python
import requests

# http://archive.ics.uci.edu/ml/datasets/Adult
DATA_INCOME = {
    'sample': 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
    'manual': 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names',
    'test': 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
}
DATA_INCOME = {
    k: requests.get(link).text
    for (k, link) in DATA_INCOME.items()
}
```

```python
from collections import OrderedDict

DATA_TYPES = OrderedDict([
    ('age', np.int8),
    ('wrkcls', np.str), # work class
    ('fnlwgt', np.int32), # final weight
    ('educt', np.str), # education
    ('eductnum', np.int8), # education num
    ('marital', np.str), # marital status
    ('occup', np.str), # occupation
    ('rltsh', np.str), # relationship
    ('race', np.str),
    ('sex', np.str),
    ('capgain', np.int32), # capital gain
    ('caploss', np.int32), # capital loss
    ('hrperw', np.int8), # hour per week
    ('ntvctry', np.str), # native country
    ('fiftyK', np.int8) # >50k
])
```

```python
# size of our data
for (k, v) in DATA_INCOME.items():
    print(k, len(v))
```

```python
from collections import OrderedDict

def get_headers():
    headers = list(DATA_TYPES.keys())
    def strip_and_replace(v):
        _v = v.strip(' .').lower()
        _v = _v.replace('-', '_')
        if _v == '?': return pd.NaT
        return _v
    
    converters = {}
    for h in headers:
        if DATA_TYPES[h] == np.str:
            converters[h] = strip_and_replace
    converters['fiftyK'] = lambda v: 1 if v == ' >50K' else 0
    return headers, converters

HEADERS, CONVERTERS = get_headers()
```

```python
import os
import tempfile
from io import StringIO

import pandas as pd

PATH_PICKLE = os.path.join(tempfile.gettempdir(), "pickles")
if not os.path.exists(PATH_PICKLE): os.mkdir(PATH_PICKLE)
PATH_DF_INCOME_TRAINING = os.path.join(PATH_PICKLE, 'income_training.pickle')
PATH_DF_INCOME_TEST = os.path.join(PATH_PICKLE, 'income_test.pickle')


def get_training_data():
    if os.path.exists(PATH_DF_INCOME_TRAINING):
        return pd.read_pickle(PATH_DF_INCOME_TRAINING)
    df = pd.read_csv(
        StringIO(DATA_INCOME['sample']), # no copy created
        header=None,
        names=HEADERS,
        na_values=['?'],
        converters=CONVERTERS,
        comment='|',
        dtype=DATA_TYPES)
    # cache it
    pd.to_pickle(df, PATH_DF_INCOME_TRAINING)
    return df

def get_test_data():
    if os.path.exists(PATH_DF_INCOME_TEST):
        return pd.read_pickle(PATH_DF_INCOME_TEST)
    df = pd.read_csv(
        StringIO(DATA_INCOME['test']),
        header=None,
        names=HEADERS,
        na_values=['?'],
        converters=CONVERTERS,
        comment='|',
        dtype=DATA_TYPES)
    # cache it
    pd.to_pickle(df, PATH_DF_INCOME_TEST)
    return df
    
DF_INCOME_TRAINING = get_training_data()
DF_INCOME_TEST = get_test_data()
```

```python
DF_INCOME_TEST.shape
```

```
(16281, 15)
```

```python
DF_INCOME_TRAINING.shape
```

```
(32561, 15)
```

### Categorical Value

Or, **discrete independent predictor** in other context, are values that can only take on integer values, while **continuous values** can take any value. In the data set we have here, columns such as *sex*, *occupation*, *relationship*, and *race*, they are all categorical values. In logistic regression, when scaling the features, we expect to normalize the features into a range [-1, 1] or [0, 1], with categorical values, this will not make sense. So the preprocess need to handle these categorical values (There exists other algorithms that can handle categorical value natively, such as **Random Forest**).

[A question on handling categorical value in LR](http://stats.stackexchange.com/questions/95212/improve-classification-with-many-categorical-variables)

Preprocess the data:

```python
# pandas.get_dummies:
#     Convert categorical variable into dummy/indicator variables
DF_INCOME_TRAINING = pd.get_dummies(DF_INCOME_TRAINING)


import os
import tempfile
PATH_PICKLE = os.path.join(tempfile.gettempdir(), "pickles")
if not os.path.exists(PATH_PICKLE): os.mkdir(PATH_PICKLE)
PATH_DF_INCOME_TRAINING = os.path.join(PATH_PICKLE, 'income_training.pickle')
PATH_DF_INCOME_TEST = os.path.join(PATH_PICKLE, 'income_test.pickle')


# less than 50K
DF_INCOME_TRAINING_GT_50K = DF_INCOME_TRAINING[DF_INCOME_TRAINING.fiftyK == 1]
# larger than 50K
DF_INCOME_TRAINING_LT_50K = DF_INCOME_TRAINING[DF_INCOME_TRAINING.fiftyK == 0]
```

### Observe data

Let's start by observing data, this is where we can apply some common sense, I think.

```python
%matplotlib inline
import matplotlib.pyplot as plt

def plot_scores_in_comparison(df1, df2, f1, f2, l1=None, l2=None,
                              xlabel=None, ylabel=None,
                              legend=True, legend_loc=None):
    plt.plot(df1[f1], df1[f2], 'o', color='orange', label=l1)
    plt.plot(df2[f1], df2[f2], 'ko', fillstyle='none', label=l2)
    xlabel = xlabel if xlabel else f1
    ylabel = ylabel if ylabel else f2
    plt.xlabel(xlabel, weight='bold')
    plt.ylabel(ylabel, weight='bold')
    plt.grid(True)
    if legend:
        plt.legend(loc=legend_loc) if legend_loc else plt.legend()
    plt.show()
```

```python
with basic_plots.zoom_plot(8, 5):
    plot_scores_in_comparison(
        DF_INCOME_TRAINING_GT_50K,
        DF_INCOME_TRAINING_LT_50K,
        'hrperw', 'eductnum',
        '>50K', '<=50K',
        'hour per week', 'years of education',
        legend_loc='lower right')
```

![png](/images/posts/4_logistic_regression_a_realistic_example_23_0.png)

```python
# The following really gives us some
# insight about the 50K group.
with basic_plots.zoom_plot(8, 5):
    plot_scores_in_comparison(
        DF_INCOME_TRAINING_GT_50K,
        DF_INCOME_TRAINING_LT_50K,
        'age', 'hrperw',
        '>50K', '<=50K',
        'age', 'hour per week')
```

![png](/images/posts/4_logistic_regression_a_realistic_example_24_0.png)

```python
from operator import itemgetter

def plot_categories(df, categories, fiftyK=True):
    '''
    Plot the num of people earns 50K+
    in their group. The group can be defined by occupation,
    country, education, race, and sex perhaps.
    
    categories
        list, list of category names.
    '''
    
    stats = {} # O(1)
    for ctgr in categories:
        total = len(df[df[ctgr] == 1])
        key = (ctgr, total)
        stats[key] = {}
        fiftyK = len(df[(df[ctgr] == 1) & (df.fiftyK == 1)])
        nfiftyK = len(df[(df[ctgr] == 1) & (df.fiftyK == 0)])
        num_fftK = stats[key].setdefault('fftk', fiftyK)
        num_nfftK = stats[key].setdefault('nfftk', nfiftyK)
    fig, ax = plt.subplots()
    
    # O(nlogn) where n == len(categories)
    keys_ordered = sorted(stats.keys(), key=itemgetter(1),
                          reverse=True)
    # O(n) where n == len(categories)
    num_fftK = [stats[k]['fftk'] for k in keys_ordered]
    num_nfftK = [stats[k]['nfftk'] for k in keys_ordered]
    
    plt.bar(
        range(len(categories)), num_fftK,
        width=0.5, color='orange', alpha=0.75, align='edge',
        label='>50K')
    plt.bar(
        range(len(categories)), num_nfftK,
        bottom=num_fftK,
        width=0.5, color='white', alpha=0.75, align='edge',
        label='<=50K')
    
    xtick_labels = [c[(c.find('_')+1):] for (c, t) in keys_ordered]
    ax.set_xticks(range(len(categories)))
    rotation = 45 if len(categories) < 20 else "vertical"
    ax.set_xticklabels(xtick_labels, rotation=rotation, fontstyle="italic")

    plt.legend(loc='upper right')
    plt.grid(True)
    plt.title("ratio of 50K people in %s groups" % len(categories))
    plt.show()
```

```python
COLUMNS = DF_INCOME_TRAINING_GT_50K.columns.tolist()
with basic_plots.zoom_plot(8, 5):
    plot_categories(
        DF_INCOME_TRAINING,
        list(filter(lambda c: c.startswith('occup'), COLUMNS)))
```

![png](/images/posts/4_logistic_regression_a_realistic_example_26_0.png)

```python
with basic_plots.zoom_plot(8, 5):
    plot_categories(
        DF_INCOME_TRAINING,
        list(filter(lambda c: c.startswith('wrkcls'), COLUMNS)))

![png](/images/posts/4_logistic_regression_a_realistic_example_27_0.png)


```python
with basic_plots.zoom_plot(8, 5):
    plot_categories(
        DF_INCOME_TRAINING,
        list(filter(lambda c: c.startswith('educt'), COLUMNS)))
```

![png](/images/posts/4_logistic_regression_a_realistic_example_28_0.png)

It looks like people with less promising education experiences (from having **no bachelor** degree to **lesser**, e.g., **hs_grad**) fall into the 50K group much less often. It's also obvious to see that most professors and PhDs make more than 50K (Though it also shows that higher eduction don't correspond to the 50K pay that strictly, I guess this might be because they are students that are not working).

```python
with basic_plots.zoom_plot(8, 5):
    plot_categories(
        DF_INCOME_TRAINING,
        list(filter(lambda c: c.startswith('ntvctry'), COLUMNS)))
```

![png](/images/posts/4_logistic_regression_a_realistic_example_30_0.png)

It looks **country** doesn't serve as a informative predictor because it takes
up too absolutely too many samples.

```python
with basic_plots.zoom_plot(4, 4):
    plot_categories(
        DF_INCOME_TRAINING,
        list(filter(lambda c: c.startswith('sex'), COLUMNS)))
```

![png](/images/posts/4_logistic_regression_a_realistic_example_32_0.png)

```python
with basic_plots.zoom_plot(6, 4):
    plot_categories(
        DF_INCOME_TRAINING,
        list(filter(lambda c: c.startswith('race'), COLUMNS)))
```

![png](/images/posts/4_logistic_regression_a_realistic_example_33_0.png)

```python
with basic_plots.zoom_plot(6, 4):
    plot_categories(
        DF_INCOME_TRAINING,
        list(filter(lambda c: c.startswith('rltsh'), COLUMNS)))
```

![png](/images/posts/4_logistic_regression_a_realistic_example_34_0.png)


```pyhton
with basic_plots.zoom_plot(6, 4):
    plot_categories(
        DF_INCOME_TRAINING,
        list(filter(lambda c: c.startswith('marital'), COLUMNS)))
```

![png](/images/posts/4_logistic_regression_a_realistic_example_35_0.png)


Somehow, people in the **never married** group doesn't fall into the 50K group that much.

Summing all the observations, I am going to experiment with an LR model with these predictors:

1. age, *age*
1. hours of work per week, *hrperw* 
1. years of education, *eductnum*
1. whether is married, *marital_never_married*
1. whether only made it to high school, *educt_hs_grad*
1. whether only made it to some college, *educt_some_college*
1. whether works in a service industry, *occup_other_service*

### Construct out model and preprocess the data

```python
from isaac.models import linear
reload(linear)
LR = linear.LogisticRegression


features = ['hrperw', 'eductnum', 'marital_never_married',
           'educt_hs_grad', 'educt_some_college', 'occup_other_service']
column_names = list(DF_INCOME_TRAINING.columns)
columns = [column_names.index(f) for f in features]


from isaac.pipeline import preprocess
# get the numpy array to work with
data = DF_INCOME_TRAINING.values
X, Y = preprocess.get_XY_from_frame(data, columns, column_names.index('fiftyK'))


model = LR.from_dimension(len(features) + 1)


# initial cost
print('cost:', model.costs(X, Y))
print('accuracy:', model.accuracy(X, Y))
```

```
cost: 1.21071031349
accuracy: 0.298117379687
```

### Start training

```python
from isaac.optimizers import gradient

descent = gradient.Descent(model, X, Y, 0.2)
descent.run(100)
print(model.costs(X, Y))
```

```
0.571466504217
```

```python
model.accuracy(X, Y)
```

```
0.75003838948435242
```

### Cross validation

```python
# Preprocess the test data
DF_INCOME_TEST = pd.get_dummies(DF_INCOME_TEST)
data_test = DF_INCOME_TEST.values
_X, _Y = preprocess.get_XY_from_frame(data_test, columns, column_names.index('fiftyK'))


print("cost:", model.costs(_X, _Y))
print("accuracy:", model.accuracy(_X, _Y))
```

```
cost: 0.350563860528
accuracy: 0.990172593821
```



    
