

import json
#import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import string
import re
import operator

from time import time
from pylab import plot,show
from scipy import stats
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix
from sklearn import decomposition
from sklearn.grid_search import GridSearchCV

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 30)

# set some nicer defaults for matplotlib
from matplotlib import rcParams

#these colors come from colorbrewer2.org. Each is an RGB triplet
dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843),
                (0.4, 0.4, 0.4)]

rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.grid'] = False
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'none'

def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecessary plot borders and axis ticks
    
    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    
    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    
    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()

input_file = open("Data/training_data2.csv",'r')
reader = csv.reader( input_file )
reader.next()

Text = []
Tags = [] 

i = 0

for line in reader:
    Text.append(str(line[1])) # original also added body ::: + " " + line[2]))
    Tags.append(str(line[3]))
    
input_file.close()

N = 30

Y_vectorizer = CountVectorizer(tokenizer = lambda x: x.split(), min_df=0, binary=True)
Y = Y_vectorizer.fit_transform(Tags)
tags = Y_vectorizer.get_feature_names()

num_posts, num_words = Y.shape
print 'number of tags in dataset = %s' % num_words
counts = Y.sum(axis=0).tolist()[0]
word_counts = list(enumerate(counts))

sorted_word_counts = sorted(word_counts, key=lambda x: x[1], reverse=True)
sorted_word_counts = map(lambda x: (tags[x[0]], float(x[1])/num_posts), sorted_word_counts)

sorted_words, sorted_counts = zip(*sorted_word_counts)

ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence
plt.figure(figsize=(15,5))
p1 = plt.bar(ind, sorted_counts[:N],  width)
plt.title('Tag Frequency in Train Set')
plt.xlabel('Top %s Tags' % N)
plt.ylabel('Tag Frequency (posts with tag/total posts)')
plt.xticks(ind+width/2., sorted_words[:N], rotation=90 )
plt.yticks(np.arange(0,.1,.01))
remove_border()
plt.show()

plt.figure(figsize=(15,5))
plt.hist(sorted_counts,bins=100)
plt.title('Tag Frequencies')
plt.xlabel('Tag Frequency (posts with tag/total posts)')
plt.ylabel('Number of Tags')
remove_border()
plt.show()



"""
    Function
    --------
    my_tokenizer(s)

    Takes input s and splits based on spaces rather than both spaces and punctuation.
    
    Parameters
    ----------
    
    s : str
        The string of input text
    
    Returns
    -------
    The split list as defined above.
    
"""
def my_tokenizer(s):
    return s.split()


"""
    Function
    --------
    makeX(Text, X_min_df)

    Converts Text to vectorized representation X
    
    Parameters
    ----------
    
    Text : list
        The list of strings of input text
    
    X_min_df : float or int
        The number corresponding to the percent of posts which we want (int for a minimum number of posts)
    
    Returns
    -------
    A tuple of the sparse COO matrix X (the vectorized representation of the text) and X_vectorizer (which accomplished the vectorizing)
    
"""
def makeX(Text,X_min_df=.0001):
    X_vectorizer = TfidfVectorizer(min_df = X_min_df, sublinear_tf=True, max_df=0.9,stop_words='english')
    X = X_vectorizer.fit_transform(Text)
    return coo_matrix(X), X_vectorizer

"""
    Function
    --------
    makeY(Tags, Y_min_df)

    Converts Tags to vectorized representation Y
    
    Parameters
    ----------
    
    Text : list
        The list of strings of input text
    
    Y_min_df : float or int
        The number corresponding to the percent of posts which we want (int for a minimum number of posts)
    
    Returns
    -------
    A tuple of the matrix Y (the vectorized representation of the tags) and Y_vectorizer (which accomplished the vectorizing)
    
"""

def makeY(Tags, Y_min_df=.0001):
    Y_vectorizer = CountVectorizer(tokenizer = my_tokenizer, min_df = Y_min_df, binary = True)
    Y = Y_vectorizer.fit_transform(Tags)
    return Y, Y_vectorizer


"""
    Function
    --------
    df_to_preds(dfmatrix, k)

    Takes input dfmatrix and returns the matrix of predicted tags.
    
    Parameters
    ----------
    
    dfmatrix : matrix
        Decision function outputs in matrix form
        
    k : int
        Number of tags to predict
    
    Returns
    -------
    The matrix of predicted tags.
    
"""
def df_to_preds(dfmatrix, k = 5):
    predsmatrix = np.zeros(dfmatrix.shape)
    for i in range(0, dfmatrix.shape[0]):
        dfs = list(dfmatrix[i])
        if (np.sum([int(x > 0.0) for x in dfs]) <= k):
            predsmatrix[i,:] = [int(x > 0.0) for x in dfs]
        else:
            maxkeys = [x[0] for x in sorted(enumerate(dfs),key=operator.itemgetter(1),reverse=True)[0:k]]
            listofzeros = [0] * len(dfs)
            for j in range(0, len(dfs)):
                if (j in maxkeys):
                    listofzeros[j] = 1
            predsmatrix[i,:] = listofzeros
    return predsmatrix

"""
    Function
    --------
    probs_to_preds(probsmatrix, k)

    Takes input probsmatrix and returns the matrix of predicted tags.
    
    Parameters
    ----------
    
    probsmatrix : matrix
        Probability outputs in matrix form
        
    k : int
        Number of tags to predict
    
    Returns
    -------
    The matrix of predicted tags.
    
"""
def probs_to_preds(probsmatrix, k = 5):
    predsmatrix = np.zeros(probsmatrix.shape)
    for i in range(0, probsmatrix.shape[0]):
        probas = list(probsmatrix[i])
        if (np.sum([int(x > 0.01) for x in probas]) <= k):
            predsmatrix[i,:] = [int(x > 0.01) for x in probas]
        else:
            maxkeys = [x[0] for x in sorted(enumerate(probas),key=operator.itemgetter(1),reverse=True)[0:k]]
            listofzeros = [0] * len(probas)
            for j in range(0, len(probas)):
                if (j in maxkeys):
                    listofzeros[j] = 1
            predsmatrix[i,:] = listofzeros
    return predsmatrix


"""
    Function
    --------
    opt_params(clf_current, params)

    Determines the set of parameters for clf_current that maximize cross-validated F1 score using sklearn's GridSearchCV
    
    Parameters
    ----------
    
    clf_current : scikitlearn clf object
        the clf we want to test
        
    params: list
        list of parameters that we will want to tune for
    
    Returns
    -------
    A tuple of the classifier's description, the score, the time it took to train, and the time it took to test.
    
"""

def opt_params(clf_current, params):
    model_to_set = OneVsRestClassifier(clf_current)
    grid_search = GridSearchCV(model_to_set, param_grid=params, score_func=metrics.f1_score)

    print("Performing grid search on " + str(clf_current).split('(')[0])
    print("parameters:")
    print(params)
    grid_search.fit(X_train, Y_train.toarray())
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    gs = grid_search.grid_scores_
    ret = [(i[0], i[1]) for i in gs]
    return best_parameters, ret


"""
    Function
    --------
    benchmark(clf_current)

    Takes the classifier passed and determines how well it performs on the test set.
    
    Parameters
    ----------
    
    clf_current : scikitlearn clf object
    
    Returns
    -------
    A tuple of the classifier's description, the score, the time it took to train, and the time it took to test.
    
"""

def benchmark(clf_current):
    print('_' * 80)
    print("Test performance for: ")
    clf_descr = str(clf_current).split('(')[0]
    print(clf_descr)
    t0 = time()
    classif = OneVsRestClassifier(clf_current)
    classif.fit(X_train, Y_train.toarray())
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    t0 = time()
    if hasattr(clf_current,"decision_function"):
        dfmatrix = classif.decision_function(X_test)
        score = metrics.f1_score(Y_test.toarray(), df_to_preds(dfmatrix, k = 5))
    else:
        probsmatrix = classif.predict_proba(X_test)
        score = metrics.f1_score(Y_test.toarray(), probs_to_preds(probsmatrix, k = 5))
        
    test_time = time() - t0

    
    print("f1-score:   %0.7f" % score)
    print("test time:  %0.3fs" % test_time)

    print('_' * 80)
    return clf_descr, score, train_time, test_time

import warnings
warnings.filterwarnings("ignore")

Y, vectorizer2 = makeY(Tags, Y_min_df=int(10))
X, vectorizer1 = makeX(Text, X_min_df=int(10))
X_current = X
X_train, X_test, Y_train, Y_test = train_test_split(X_current,Y)

results = []
'''
classlist = [
(Perceptron(), {'estimator__penalty': ['l1', 'elasticnet'],"estimator__alpha":[.001,.0001],'estimator__n_iter':[50]}), 
(PassiveAggressiveClassifier(), {'estimator__C':[.01,.1,1.0],'estimator__n_iter':[50]}),
(LinearSVC(), {'estimator__penalty': ['l1','l2'], 'estimator__loss': ['l2'],'estimator__dual': [False], 'estimator__tol':[1e-2,1e-3]}),
(SGDClassifier(), {'estimator__penalty': ['l1', 'elasticnet'],"estimator__alpha":[.0001,.001],'estimator__n_iter':[50]}),
(MultinomialNB(), {"estimator__alpha":[.01,.1],"estimator__fit_prior":[True, False]}),
(BernoulliNB(), {"estimator__alpha":[.01,.1],"estimator__fit_prior":[True, False]})
            ]
'''
classlist = [(SGDClassifier(), {'estimator__penalty': ['l1', 'elasticnet'],"estimator__alpha":[.0001,.001],'estimator__n_iter':[50]}), (LinearSVC(), {'estimator__penalty': ['l1','l2'], 'estimator__loss': ['l2'],'estimator__dual': [False], 'estimator__tol':[1e-2,1e-3]})]

for classifier, params_to_optimize in classlist:
    best_params, gs = opt_params(classifier, params_to_optimize)
    results.append(benchmark(best_params['estimator']))


# make some plots
def plot_results(current_results, title = "Score"):
    indices = np.arange(len(current_results))
    
    results2 = [[x[i] for x in current_results] for i in range(4)]
    
    clf_names, score, training_time, test_time = results2
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(training_time)
    
    plt.figure(1, figsize=(14,5))
    plt.title(title)
    plt.barh(indices, score, .2, label="score", color='r')
    plt.barh(indices + .3, training_time, .2, label="training time", color='g')
    plt.barh(indices + .6, test_time, .2, label="test time", color='b')
    plt.yticks(())
    plt.legend(loc='best')
    plt.xlabel('Mean F1 Score (time values are indexed so max training time = 1.0)')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)
    
    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)
    
    plt.show()
    
plot_results(results, title = "Classifier F1 Results on Tf-idf vector")


def topic_transform(X, n_topics = 10, method = "SVD"):
    if (method == "NMF"):
        topics = decomposition.NMF(n_components=n_topics).fit(X)
    else:
        topics = decomposition.TruncatedSVD(n_components=n_topics).fit(X)
    X_topics = topics.transform(X)
    for i in range(0,X_topics.shape[0]):
        theline = list(X_topics[i])
        # following line is only important for SVD
        theline = [(x * int(x > 0)) for x in theline]
        topic_sum = np.sum(theline)
        X_topics[i] = list(np.divide(theline,topic_sum))
    return X_topics, topics

def print_topics(topics,vectorizer1,n_top_words = 12):
    # Inverse the vectorizer vocabulary to be able
    feature_names = vectorizer1.get_feature_names()
    
    for topic_idx, topic in enumerate(topics.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()


#X_nmf, nmf = topic_transform(X, n_topics = 20,method = "NMF")
#print_topics(nmf,vectorizer1)

#X_svd, svd = topic_transform(X, n_topics = 20,method = "SVD")
#print_topics(svd,vectorizer1)

nmf_results = []
svd_results = []

classif_tuple = SGDClassifier(), {'estimator__penalty': ['l1'],"estimator__alpha":[.0001,.001],'estimator__n_iter':[50]}

num_topics_list = [100,200]
'''
print "++++++++++++"
print "testing SGD classifier with topics --"
for num_topics in num_topics_list:
    print 'Now testing for NMF with ' + str(num_topics) + ' topics'
    X_nmf, nmf = topic_transform(X, n_topics = num_topics, method = "NMF")
    X_current = X_nmf
    X_train, X_test, Y_train, Y_test = train_test_split(X_current,Y)
    nmf_results.append(benchmark(SGDClassifier(penalty = 'l1',alpha = .0001, n_iter = 50))[1])

    
    print 'Now testing for SVD with ' + str(num_topics) + ' topics'
    X_svd, svd = topic_transform(X, n_topics = num_topics, method = "SVD")
    X_current = X_svd
    X_train, X_test, Y_train, Y_test = train_test_split(X_current,Y)
    svd_results.append(benchmark(SGDClassifier(penalty = 'l1',alpha = .0001, n_iter = 50))[1])
'''


