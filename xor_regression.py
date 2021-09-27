import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pickle
import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42



windowsize = 10
n_neurons = 1_000
len_trial = 260.
with open("xor_match_responses_dc05.pkl", 'rb') as f:
    match_responses = pickle.load(f)

with open("xor_nomatch_responses_dc05.pkl", 'rb') as f:
    nomatch_responses = pickle.load(f)

with open("xor_match_block_dc05.pkl", 'rb') as f:
    match_block = pickle.load(f)

with open("xor_nomatch_block_dc05.pkl", 'rb') as f:
    nomatch_block = pickle.load(f)


n_trials = 400
n_splits = 200
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)
#logisticRegr = LogisticRegression(max_iter=1000)
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

        #logisticRegr.fit(X_train, y_train)
        clf.fit(X_train,y_train)
        #scores[idx,cnt] = logisticRegr.score(X_test, y_test)
        clfscores[idx, cnt] = clf.score(X_test, y_test)
        cnt +=1

columns=['\u03C4-10',
         '\u03C4+10',
         '\u03C4+20',
         '\u03C4+30',
         '\u03C4+40',
         '\u03C4+50',]

df_clfscores = pd.DataFrame(clfscores.T, columns=columns)

plt.figure(figsize=(8,6),dpi=300)

sns.violinplot(data = df_clfscores,showmeans=True,
               showextrema=True, showmedians=True,
               palette="PuRd",alpha=0.5)
sns.stripplot(data=df_clfscores,size=2, color="k", alpha=0.3,jitter=0.1)

plt.title('XOR → Reservoir → Support Vector Machine')
plt.ylabel('Accuracy')
plt.xlabel('Time (ms)')

plt.axhline(0.5,linestyle='--',color = 'k')
plt.axvline(0.5,linestyle='--',color = 'k')
plt.ylim([0.3, 1])

plt.savefig('Accuracy XOR_dc05.pdf',bbox_inches='tight')

mpl.rcParams['font.size'] = 10

plt.figure(figsize=(6,4),dpi=300)
sns.heatmap(match_block[:,:,300],cmap='Greys',cbar=False,yticklabels=False)
plt.xlim([0,201])
plt.xticks(np.arange(0,201,10),labels=np.arange(0,201,10))
plt.axvline(100,linewidth=2,color = '#F53A61')
plt.axvline(150,linewidth=2,color = '#F53A61')

for i in np.arange(140,200,10):
    plt.axvline(i,linewidth=1,color = '#0729F5')

plt.axhline(400,linewidth=1,color = 'k',linestyle='--')
plt.axhline(900,linewidth=1,color = 'k',linestyle='--')

plt.axhline(800,linewidth=2,color = 'k')
plt.text(-10,400,'excitatory',rotation='vertical',
         verticalalignment='center')
plt.text(-10,900,'inhibitory',rotation='vertical',
         verticalalignment='center')
plt.text(154,-30,'response time',
         verticalalignment='center')
plt.text(108,-30,'stimulation',
         verticalalignment='center')
plt.text(43,-30,'baseline',
         verticalalignment='center')
plt.xlabel('time (ms)')
plt.savefig('spikes XOR_match_both_dc05.pdf',bbox_inches='tight')
