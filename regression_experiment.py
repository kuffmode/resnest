import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pickle

windowsize = 10
n_neurons = 1_000
len_trial = 260.
with open("match_responses.pkl", 'rb') as f:
    match_responses = pickle.load(f)

with open("nomatch_responses.pkl", 'rb') as f:
    nomatch_responses = pickle.load(f)

n_trials = 100
n_splits = 100
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)
logisticRegr = LogisticRegression(max_iter=1000)
clf = make_pipeline(StandardScaler(),SVC(kernel='sigmoid'))

scores = np.zeros((6,n_splits))
clfscores = scores.copy()
for idx, window in enumerate(np.arange(0,60,windowsize)):

    X = np.zeros((n_neurons,n_trials*2))
    Y = np.zeros(n_trials*2)
    for trial in range(n_trials):

        X[:,trial] = np.sum(match_responses[:,window:window+windowsize,trial]>0,axis=1)
        X[:,trial+n_trials] = np.sum(nomatch_responses[:,window:window+windowsize,trial]>0,axis=1)

        Y[trial] = 0
        Y[trial+n_trials] = 1

    X = X.T
    cnt = 0
    for train_index, test_index in sss.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        logisticRegr.fit(X_train, y_train)
        clf.fit(X_train,y_train)
        scores[idx,cnt] = logisticRegr.score(X_test, y_test)
        clfscores[idx, cnt] = clf.score(X_test, y_test)
        cnt +=1

plt.violinplot(scores.T,showmeans=True, showextrema=True, showmedians=True)
plt.violinplot(clfscores.T,showmeans=True, showextrema=True, showmedians=True)

plt.axhline(0.5,color = 'r')
plt.savefig('accuracy.png')