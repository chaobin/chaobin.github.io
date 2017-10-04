---
layout: post_mathjax
type: post
title: "A Probablistic Approach in Pattern Recognition and Bayes' Theorem"
tags:
    - bayes theorem
    - machine learning
    - text classification
description: In supervised learning, data is provided to us which can be considered as evidence. For example, in a text classification system, we may have a collection of texts (corpus) that can be percieved as evidence as to how language is used in real world that can give us insight to the text genre, author gender, text sentiment, etc. And based on these evidence, we can try to get a better opinion to classify a new text.
---

In supervised learning, data is provided to us which can be considered as
evidence. For example, in a text classification system, we may have a collection
of texts (corpus) that can be percieved as evidence as to how language is used
in real world that can give us insight to the text genre, author gender, text
sentiment, etc. And based on these evidence, we can try to get a better opinion
to classify a new text.

In a website where user can post articles and upload comments, we are tasked to
identify language abuse in order to folster a friendly online environment. We
have been given *10,000* comments that are labeled to help develop a model.
Suppose we already know *1* out of every *5* comments contains language abuse,
without knowing anything else at this point, our model classifies **all comments
to be non-abusive**, as a result of minimizing the classification error. Since
now we have *10,000* comments to look at as evidence, we should be able to do
better than this pure guesswork. So we seek to answer a question, that given the
evidence, what is our **renewed belief** as to whether this text is abusive.

## Bayes' Theorem - a way to quantify the updated confidence on an event based on new evidence

The definition of **conditional probability** is this:


{% raw %}
<div class="equation" data="P(A|B) = \frac{P(A \cap B)}{P(B)}"></div>
{% endraw %}


By the same definition, we also have this:


{% raw %}
<div class="equation" data="P(B|A) = \frac{P(A \cap B)}{P(A)}"></div>
{% endraw %}


Using algebra, substituting *P(A∩B)* in the first definition using the second
definition, we have:


{% raw %}
<div class="equation" data="P(A|B) = \frac{P(B|A)P(A)}{P(B)}"></div>
{% endraw %}


The above is referred to as **Bayes' Theorem** named after the English
statistician [Thomas Bayes](https://en.wikipedia.org/wiki/Thomas_Bayes). A
critical role this theorem plays in the task of estimating a conditional
probability is it **reexpresses** the conditional probability with another
conditional probability that is often easier to calculate.

## Maximum likelihood - after seeing the evidence, which is the most likely event

Continuing our imaginary task of identifying language abuse, suppose we define:


{% raw %}
<div class="equation" data=" P(C_{a}) "></div>
{% endraw %}


as the event that the text being classified is **a**busive, and:


{% raw %}
<div class="equation" data=" P(C_{n}) "></div>
{% endraw %}


as the event that the text being classified is **n**on-abusive. And we have:


{% raw %}
<div class="equation" data=" 1 = P(C_{a}) + P(C_{n}) "></div>
{% endraw %}


Using Bayes' Theorem, we can calculate two conditional probabililites:


{% raw %}
<div class="equation" data=" P(C_{a}|t) = \frac{P(t|C_{a})P(C_{a})}{P(t)} "></div>
{% endraw %}


and


{% raw %}
<div class="equation" data=" P(C_{n}|t) = \frac{P(t|C_{n})P(C_{n})}{P(t)} "></div>
{% endraw %}


where **t** is the text being classified. The calculations above can be
interpreted as saying, given evidence **t**, what is our **updated belief** that
**t** is abusive or non-abusive depending on which calculation we are
performing. We then **compare** the result of two calculcations (in a case with
more than 2 classes, we can perform calculations with each class.), and classify
the text to the class for which we have the **largest value** of the class-
conditional probability.

Since **P(t)** is class-independant in estimations for all classes, and it is
the comparative differences between estimations that determine the result of
classification, we can drop the **P(t)** in all calculations for each class, and
we have a **discriminant function**:


{% raw %}
<div class="equation" data=" P(C_{i}|t) = P(t|C_{i})P(C_{i}) "></div>
{% endraw %}



This same observation also allows us to further transform our discriminant
function using any **monotonic** function, because the magnitude of differences
will be preserved. A logarithm is an obvious one to first try because in
addition to being monotonic, it transforms the multiplication into addition:


{% raw %}
<div class="equation" data="P(C_{i}|t) = ln(P(t|C_{i})) + ln(P(C_{i})) "></div>
{% endraw %}
 
 
## Spam classifier - let's put all of these into code 

**In [27]:**

{% highlight python %}
# We are using numpy for computation, and Python NLTK for some utilities 

# A word on NLTK
# NLTK perhaps remains the go-to recommendations for many people when it comes to NLP.
# There are a couple of things people liked about this library:
# 1. It has an abundant collection of corpora, installed with nltk.download()
# 2. It has an abundant collection of algorithms for many NLP routines,
#   such as tokenization, stemming, lemmatization, etc.
# 3. It supports a lot of languages.
#
# However, NLTK falls short when it comes to performance. If you look at
# the source code implementation of algorithms such as that of some
# tokenizers in nltk, those are pure Python code.
# When we are processing a large amount of text, or that our model is
# training with online data when data has to be feed-in and
# processed in real time, the heavy cost of pure Python
# code easily becomes bottleneck. With text processing,
# a lot of str objects are being created and manipulated.
# These operations have extensive memory footprints, also because str objects
# are immutable in Python, the many intermediate operations over
# a str objects cast away the the object soon it finishes off with it,
# causing the garbage collector to be very busy while
# keeps the system busy alocating memory in the heap.
#
# For this reason, efforts such as spacy (https://spacy.io/) emerged.
# Spacy is implemented in Cython, and claims to be the-fastest
# in several preprocessing algorithms. The point here is
# that even though Python may be the most popular choice in NLP,
# one should be aware of Python's shortcomings and
# also know that people developed solutions for it.
{% endhighlight %}

**In [28]:**

{% highlight python %}
import numpy as np
import nltk
print("numpy: ", np.__version__)
print("nltk: ", nltk.__version__)
{% endhighlight %}

    numpy:  1.11.0
    nltk:  3.2.2


**In [29]:**

{% highlight python %}
# labeled email samples from http://csmining.org/index.php/spam-email-datasets-.html
import os
path_emails = '/Users/cbt/Projects/isaac/data'
path_label = os.path.join(path_emails, 'SPAMTrain.label')
IS_SPAM = 0
NO_SPAM = 1
{% endhighlight %}
 
The following is the preprocessing that converts the *.eml* files to plain text
files. Note that the conversion ignored all properties of an email file and only
used the subject and the body. *(There are very good reasons to make use of
properties of an email file such as sender IP address. It shouldn't be
surprising that an industry spam classifier, such as that of Gmail, to make use
of many properties of an email in feature engineering, and even very likely have
an [ensemble of models](https://en.wikipedia.org/wiki/Ensemble_learning))* 

**In [30]:**

{% highlight python %}
import glob

import mailparser

def extract_text_from_email(eml_file):
    mail = mailparser.parse_from_file(eml_file)
    return os.linesep.join([mail.subject, mail.body])

def parse_emails(src, saveto):
    emls = glob.glob(os.path.join(path_emls, '*.eml'))
    texts = ((f, extract_text_from_email(f)) for f in emls)
    for (f, text) in texts:
        name = os.path.basename(f)
        name, ext = os.path.splitext(name)
        name = '%s.msg' % name
        dest = os.path.join(saveto, name)       
        with open(dest, 'w') as out:
            out.write(text)
{% endhighlight %}

**In [31]:**

{% highlight python %}
# a set utility functions
def head(sequence_or_map):
    if isinstance(sequence_or_map, list):
        return sequence_or_map[0]
    elif isinstance(sequence_or_map, dict):
        return next(iter(sequence_or_map.items()))
{% endhighlight %}
 
## Naive Bayes - a word on how to encode a text into our discriminant function

Recall that our discrimimnant function looks like this:


{% raw %}
<div class="equation" data=" P(C_{i}|t) = ln(P(t|C_{i})) + ln(P(C_{i})) "></div>
{% endraw %}


The **P(C)** can be derived out of the data by simply counting the members of
each class/group and normalize it by the total number of samples. This is
sometimes called **prior** *(actually, determining prior is of greater
importance and requires further reasoning. I am leaving this out of the scope of
this article, only pointing out that we are drawing our prior out of our
training data set)*. The **P(t)** is what we are examining right now. I feel
inclined to direct our discussion to a more intuitive level at this point, that
in our text classification task, using the discriminant function we have devised
from the Bayes' Theorem, we seek to update our prior **P(C)** with evidence
**t** in the hope that the evidence improves the guesswork we would otherwise
have to rely solely on the prior. So now we ask, how do we encode our evidence?
Intuitively, a spam email is usually observed to do several things:

1. such as encouraging reader to click an email,
1. or faking a story in order to trick the reader to return an email that
contains important information,
1. or just a marketing email that bombards the reader with product information,
1. etc.

Authoring emails like this can usually result in a pattern of choosing words
(such as "please click LINK", "or the service is free"). So the encoding scheme
we seek to encode our evidence is to capture such intuition. To begin with, we
can think of the word frequency. For example, suppose in the *1000 (500 spams
and 500 non-spam emails)* emails we collected and labeled, we calculate the word
frequency like this:

1. the vocabulary **V** is defined by unioning all the words in all emails.
1. frequency of a word is defined by the total number of occurences of it in all
emails (in certain group)

The computation complexity of the above process is linear to the amount of
training text and the average length of them. After this process, we then come
to answer the next question in line, how do we encode the text **t** that is
being classified and is our evidence? In principle, if we want to encode a text
as a distribution of words, there are certain things we need to take into
account. The sentence "How are you" is a deterministic combination of words
because of the English grammar, so the probablistic encoding should be:


{% raw %}
<div class="equation" data='P("how")P("are"|"how")P("you"|("how", "are"))'></div>
{% endraw %}


So, in order for our encoding of the text to quality as a probablistic
distribution of words, our encoding is very difficult to calculate because of
the dependency of one word on another or on others.

An assumption was made by people who first tried to apply Bayes' Theorem in text
classification, that one word does not depend on another in their appearance in
texts. By the definition of conditional probability:


{% raw %}
<div class="equation" data="P(A|B) = \frac{P(A \cap B)}{P(B)} "></div>
{% endraw %}


{% raw %}
<div class="equation" data="P(A|B) = \frac{P(A)P(B|A)}{P(B)}"></div>
{% endraw %}


If we apply that assumption above into the above definition, we have:


{% raw %}
<div class="equation" data="P(A|B) = \frac{P(A \cap B)}{P(B)} "></div>
{% endraw %}


{% raw %}
<div class="equation" data="P(A|B) = \frac{P(A)P(B)}{P(B)}"></div>
{% endraw %}


{% raw %}
<div class="equation" data="P(A|B) = P(A) "></div>
{% endraw %}


note how the dependant **P(A)** is assumed away above. Back to our "How are you"
example, we then have:


{% raw %}
<div class="equation" data='P("how")P("are")P("you")'></div>
{% endraw %}


This assumption is refered to as **Naive Bayes**, where the word "naive" means
simplified, and the word "bayes" refers to the Bayes' Theorem.

Now we can finally revisit our initial question as to how to encode our evidence
**t**. With calculated word frequencies, the **t** is encoded as a frequency
distribution of words (word that are not in the vocabulary **V** is not
accounted). Intuitively, a spam-email-potential likely contains words that show
a lot in the spam emails, and does so proportionally more obviously than in the
non-spam emails.

We are now ready to code. 

**In [32]:**

{% highlight python %}
# read each email into words and do the following:
# - remove stopwords
# - remove punctuation
# - possibly remove high frequency words

import re
import glob
import codecs
import string
import math
from collections import Counter

from bs4 import BeautifulSoup
from nltk.tokenize.casual import casual_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer


STOPWORDS = set(stopwords.words('english'))
EXTRA_STOPWORDS = set(["i'm", "even", "it's", "using", "also", "don't", "isn't",
                      "i've", "within", "without"])
STOPWORDS = set(STOPWORDS) | EXTRA_STOPWORDS
PUNCT = set(list(string.punctuation))
REMOVED_WORDS = (STOPWORDS | PUNCT)
REMOVED_WORDS = (REMOVED_WORDS | set(list(string.ascii_lowercase)) | set(list(string.digits)))
REMOVED_WORDS |= set(['...', '..', ])


from nltk.stem.api import StemmerI
from nltk.compat import python_2_unicode_compatible


@python_2_unicode_compatible
class SimpleStemmer(StemmerI):
    
    IRREGULAR_FORMS = {
        'email': ['e-mail']
    }
    
    def __init__(self):

        self.pool = {}
        for word in self.IRREGULAR_FORMS:
            for form in self.IRREGULAR_FORMS[word]:
                self.pool[form] = word

    def stem(self, word):
        return self.pool.get(word, word)

    def __repr__(self):
        return '<%s>'.format(self.__class__.__name__)

LEMMATIZER = WordNetLemmatizer()
STEMMERS = [SnowballStemmer('english'), SimpleStemmer()]

class Vocabulary(Counter):
    
    def size(self):
        return len(self)
    
    def drop(self, threshold=None, keep=(), remove=(), pattern=None):
        if threshold:
            try:
                lt, gt = threshold
            except ValueError as e:
                raise ValueError('threshold must be a 2-item tuple, %s given' % str(threshold))
        else:
            lt, gt = -1, math.inf
            
        deletion = []
        for (k, count) in self.items():
            if (
                (count <= lt or count >= gt) or
                (k in remove) or
                (pattern and pattern.match(k))
            ) and not (k in keep):
                deletion.append(k)
        size = self.size()
        for k in deletion:
            del self[k]
        print('purged %d word' % (size - self.size()))
    
    def search(self, pattern):
        matches = []
        for (word, count) in self.items():
            if pattern.match(word):
                matches.append(word)
        return matches
    
def load(filename):
    with codecs.open(filename, "r",
                     encoding='utf-8', errors='replace') as raw:
        text = raw.read()
    return text

def read_raw():
    '''
    The reason this is used over the corpus reader in NLP
    is to allow dealing with unicode error.
    '''
    texts = {}
    files = glob.glob('%s/*.msg' % path_emails)
    for fname in files:
        text = load(fname)
        texts[os.path.basename(fname)] = text
    return texts

def read_labels():
    labels = {}
    with open(path_label) as label_file:
        for line in label_file:
            int_label, filename = line.split(' ')
            filename = filename.strip()
            filename = filename.replace('eml', 'msg')
            labels[filename] = int(int_label)
    return labels

def remove_stopwords(words):
    _words = []
    
    for word in words: # O(n)
        _word = word.lower()
        if not _word in REMOVED_WORDS: # O(1)
            _words.append(_word)
    return _words

def clean_html(text):
    soup = BeautifulSoup(text, 'html5lib')
    return soup.get_text()

def stem(word):
    _word = word
    for stemmer in STEMMERS:
        _word = stemmer.stem(word)
    return _word

def tokenize(text):
    # some sanitization
    text = clean_html(text)
    # tokenize, preserving URL
    # There are a variety of tokenizers in NLP, and there is no "BEST" tokenizer.
    # A choice is usually made considering the specific application.
    words = [stem(LEMMATIZER.lemmatize(w.lower()))
             for w in casual_tokenize(text, preserve_case=False)
             if not w.lower() in set()]
    # remove stopwords and some punctuations (not all)
    words = remove_stopwords(words)
    return words

def parse_emails(texts):
    named_emails = []
    for (fname, text) in texts.items():
        words = tokenize(text)
        named_emails.append((fname, words))
    print('%d emails parsed' % len(named_emails))
    return named_emails

def merge(parsed_emails, labels):
    data = []
    for (name, words) in parsed_emails:
        data.append((words, labels[name]))
    return data

{% endhighlight %}

**In [33]:**

{% highlight python %}
from collections import Counter


class ModelUntrained(Exception): pass
class ModelNotReady(Exception): pass
class Classifier(object):
    
    def __init__(self):
        self.likelihood = None
        self.num_classes = None
        self.priors = None
        
        self.V_C = None
        self.V = None
        
        self.num_trainings = 0
        self.distribution = {}
        
        self._prior_updated = False
        self._likelihood_updated = False
    
    def stable(self):
        return all((
                self._prior_updated, self._likelihood_updated))
    
    def require_update(self):
        self._prior_updated = self._likelihood_updated = False
    
    def update_vocabulary(self, training_set, drop_threshold=None,
                          keep=(), remove=(), pattern=None):
        self.num_trainings += len(training_set)
        label_count = {}
        # initialize vocabulary
        self.V_C = self.V_C or {}
        self.V = self.V or Vocabulary()

        # build vocabulary of all classes and of each class
        for (words, label) in training_set:
            v_c = self.V_C.setdefault(label, Vocabulary())
            v_c.update(words)
            self.V.update(words)
            self.distribution[label] = (self.distribution.setdefault(label, 0) + 1)
        
        for (_, v_c) in self.V_C.items():
            v_c.drop(threshold=drop_threshold, keep=keep, remove=remove, pattern=pattern)
        self.V.drop(threshold=drop_threshold, keep=keep, remove=remove, pattern=pattern)
        print('updated vocabulary size:', self.V.size())
        self.require_update()
    
    def update_prior(self):
        self.priors = self.priors or {}
        # calculate the priors
        for (label, count) in self.distribution.items():
            self.priors[label] = math.log(count / self.num_trainings)
        self.num_classes = len(self.priors)
        self._prior_updated = True

    def update_likelihood(self):
        # calculate the probability of word in class-vocabulary
        # against all-vocabulary
        self.likelihood = self.likelihood or {}
        for (c, v_c) in self.V_C.items():
            likelihood = self.likelihood.setdefault(c, {})
            for (word, count) in v_c.items():
                # Lapalace smoothing (additive smoothing)
                p = (count + 1) / (v_c.size() + self.V.size())
                likelihood[word] = math.log(p)
        self._likelihood_updated = True
                
    def update(self):
        self.update_prior()
        self.update_likelihood()
    
    def preprocess(self, filename):
        text = load(filename)
        words = tokenize(text)
        return words
    
    def train(self, training_set):
        self.update_vocabulary(training_set)
        self.update()
        
    def classify(self, words):
        if not self.stable(): raise ModelNotReady()
        if isinstance(words, str): # filename
            words = self.preprocess(words)
        probabilities = np.zeros(
            self.num_classes,
            dtype=np.float32)
        classes = []
        for (i, (label, v_c)) in enumerate(self.likelihood.items()):
            classes.append(label)
            prior = self.priors[label]
            probabilities[i] = prior
            for word in words:
                p = v_c.get(word)
                if p is None: continue
                probabilities[i] += (probabilities[i] + p)
        return classes[probabilities.argmax()]
    
{% endhighlight %}
 
We define a testing utility as cross validation using the splitted data set
which itself is preprocessed. 

**In [34]:**

{% highlight python %}
def cross_validate(testing_set, labels, classifier):
    error = 0
    for (i, (name, words)) in enumerate(testing_set):
        label = labels[name]
        prediction = classifier.classify(words)
        if label != prediction:
            error += 1
    return error / len(testing_set)
{% endhighlight %}

**In [35]:**

{% highlight python %}
# The I/O operation that loads data into Python
texts = read_raw()
labels = read_labels()
{% endhighlight %}

**In [36]:**

{% highlight python %}
# The preprocessing of all training data
# This is the most computation-expensive part of the process. But this can be carried out
# offline.
parsed_emails = parse_emails(texts)
{% endhighlight %}

    4327 emails parsed

 
Splitting the data set into a training data set and a testing data set. 

**In [37]:**

{% highlight python %}
# split the training set
SPLIT = int(len(parsed_emails) * (0.9))
training_set = parsed_emails[:SPLIT]
testing_set = parsed_emails[SPLIT:]
{% endhighlight %}
 
The *update_vocabulary* is an abstration to allow us fine-tune the model using
observations about the model. Basically, it offers the flexibility to the caller
to do one thing, to remove a subset of vocabulary. The flexibility includes:
1. a regular expression pattern
1. a set
1. a threshold of frequency. words that have a frequency lower than the lower
bound or higher than the upper bound will be discarded. 

**In [38]:**

{% highlight python %}
PTN_REMOVE = re.compile(r'_+|^\w\d$|^\d+px$|^#\w+$')
REMOVE = set(('wa', '�', '=d', 'font-size', 'ha', 'color', 'br',
         'text', 'text-decoration', 'de', 'font-weight',
         'content-transfer-encoding', 'le', 'content-type',
         'sans-serif', '-8859-1', '•', 'bb',
         'quoted-printable', 'font-family', 'plain',
         'charset', 'spam', 'ba', 'dqo', 'bc', 'ab',
         'af', '©', 'line-height', '..', 'email', ';}', 'year', 'list'))
classifier = Classifier()
classifier.update_vocabulary(merge(training_set, labels),
                             drop_threshold=(10, math.inf),
                             keep=('free',),
                             remove=REMOVE,
                             pattern=PTN_REMOVE
                            )
classifier.update()
{% endhighlight %}

    purged 39376 word
    purged 41881 word
    purged 70627 word
    updated vocabulary size: 7230


**In [39]:**

{% highlight python %}
cross_validate(testing_set, labels, classifier)
{% endhighlight %}




    0.8660508083140878


 
Our model reports a near *87%* success rate, and this is not too bad. Now we can
take a look at the model and see the usage of word in different categories: 

**In [41]:**

{% highlight python %}
# The word usage in the spams:
classifier.V_C[IS_SPAM].most_common(100)
{% endhighlight %}




    [('click', 1031),
     ('free', 1015),
     ('price', 931),
     ('state', 919),
     ('please', 876),
     ('new', 828),
     ('get', 820),
     ('one', 816),
     ('time', 769),
     ('business', 673),
     ('address', 656),
     ('information', 645),
     ('people', 643),
     ('order', 638),
     ('money', 622),
     ('united', 606),
     ('right', 606),
     ('name', 590),
     ('may', 589),
     ('first', 570),
     ('receive', 563),
     ('unsubscribe', 524),
     ('make', 515),
     ('company', 507),
     ('service', 502),
     ('message', 502),
     ('report', 501),
     ('government', 496),
     ('day', 488),
     ('like', 486),
     ('2009', 466),
     ('many', 461),
     ('home', 461),
     ('offer', 454),
     ('program', 441),
     ('site', 437),
     ('would', 434),
     ('change', 433),
     ('want', 429),
     ('newsletter', 412),
     ('see', 412),
     ('send', 412),
     ('use', 411),
     ('today', 401),
     ('view', 401),
     ('go', 394),
     ('mailing', 393),
     ('best', 391),
     ('work', 388),
     ('month', 387),
     ('link', 379),
     ('need', 359),
     ('form', 359),
     ('internet', 355),
     ('number', 352),
     ('call', 350),
     ('online', 350),
     ('product', 348),
     ('world', 342),
     ('sent', 340),
     ('system', 337),
     ('island', 335),
     ('million', 334),
     ('life', 332),
     ('web', 324),
     ('page', 320),
     ('professional', 318),
     ('help', 318),
     ('city', 317),
     ('way', 314),
     ('news', 298),
     ('contact', 297),
     ('made', 297),
     ('policy', 295),
     ('arial', 294),
     ('privacy', 289),
     ('find', 287),
     ('take', 286),
     ('country', 284),
     ('wish', 283),
     ('every', 282),
     ('software', 281),
     ('two', 279),
     ('subject', 279),
     ('none', 278),
     ('rate', 273),
     ('removed', 269),
     ('grant', 268),
     ('kingdom', 264),
     ('phone', 256),
     ('member', 255),
     ('100', 253),
     ('much', 250),
     ('since', 250),
     ('lincoln', 250),
     ('special', 249),
     ('dollar', 248),
     ('back', 247),
     ('used', 246),
     ('visit', 245)]



**In [42]:**

{% highlight python %}
classifier.V_C[NO_SPAM].most_common(100)
{% endhighlight %}




    [('wrote', 1990),
     ('unsubscribe', 1907),
     ('file', 1617),
     ('use', 1581),
     ('one', 1562),
     ('get', 1505),
     ('time', 1288),
     ('new', 1258),
     ('would', 1246),
     ('like', 1234),
     ('user', 1221),
     ('work', 1164),
     ('problem', 1138),
     ('2010', 1134),
     ('subject', 1070),
     ('2002', 1038),
     ('system', 1010),
     ('may', 990),
     ('message', 983),
     ('linux', 957),
     ('contact', 908),
     ('archive', 895),
     ('make', 880),
     ('need', 870),
     ('know', 859),
     ('way', 841),
     ('version', 832),
     ('trouble', 831),
     ('doe', 805),
     ('see', 796),
     ('could', 781),
     ('mailing', 772),
     ('group', 770),
     ('think', 765),
     ('listmaster@lists.debian.org', 763),
     ('package', 757),
     ('debian', 756),
     ('kde', 752),
     ('change', 742),
     ('want', 729),
     ('server', 717),
     ('window', 704),
     ('people', 701),
     ('thing', 700),
     ('first', 691),
     ('debian-user-request@lists.debian.org', 675),
     ('said', 644),
     ('help', 637),
     ('web', 634),
     ('say', 599),
     ('update', 592),
     ('kernel', 582),
     ('find', 581),
     ('information', 581),
     ('size', 581),
     ('still', 573),
     ('since', 563),
     ('go', 563),
     ('something', 562),
     ('run', 559),
     ('software', 556),
     ('right', 553),
     ('well', 551),
     ('issue', 551),
     ('set', 546),
     ('good', 545),
     ('much', 545),
     ('look', 530),
     ('code', 529),
     ('line', 528),
     ('subscription', 525),
     ('url', 522),
     ('really', 518),
     ('error', 515),
     ('bug', 512),
     ('used', 505),
     ('date', 503),
     ('free', 499),
     ('please', 495),
     ('root', 492),
     ('apr', 488),
     ('try', 487),
     ('support', 478),
     ('etc', 478),
     ('many', 476),
     ('read', 476),
     ('pm', 467),
     ('mail', 466),
     ('back', 466),
     ('thanks', 456),
     ('network', 452),
     ('two', 451),
     ('day', 442),
     ('point', 441),
     ('home', 436),
     ('case', 436),
     ('sure', 430),
     ('take', 427),
     ("can't", 427),
     ('install', 422)]


