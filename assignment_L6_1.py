# -*- coding: utf-8 -*-
# Zadanie 1 (7 pkt.)
"""
Kod muszą państwo zaimplementować w pliku `assignment_L6_1.py`, a gotowe zadanie oddajemy wypychając zmiany na repozytorium.
"""


from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from sklearn import model_selection
from sklearn.cluster import KMeans
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
from sklearn.feature_selection import SelectKBest
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from itertools import product
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB




iris = load_iris()

X = iris.data
y = iris.target
feature_names = iris.feature_names
# XSepalLength = X[:,0]
# YSepalWidth = X[:,1]

skb = SelectKBest(k='all')
skb.fit(X, y)
X_vec = skb.transform(X)

from sklearn.feature_selection import mutual_info_classif


feature_scores = mutual_info_classif(X_vec, y)

print 'Dwa najlepsze atrybuty to {0}, {1}'.format(*sorted(zip(feature_scores, feature_names), reverse=True))
# /\ Odpowiedź dla zadania pierwszego | zadanie drugie \/ 

"""
+ Załaduj zbiór danych __iris__ korzystając z funkcji [load_iris](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
+ Korzystając z funkcji [SelectKBest](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html) oraz kryterium [mutual_info_classif](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn.feature_selection.mutual_info_classif) wybierz najlepsze __dwa__ atrybuty 
"""

X = iris.data[:, [0, 2]]
y = iris.target

# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=10)
clf2 = KNeighborsClassifier(n_neighbors=1)
clf3 = SVC(kernel='rbf', probability=True)
clf4 = LinearSVC()
clf5 = GaussianNB()

clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
clf4.fit(X, y)
clf5.fit(X, y)

f, axarr = plt.subplots(3, 2, sharex='col', sharey='row', figsize=(10, 8))

# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))


for idx, clf, tt in zip(product([0, 1, 2], [0, 1]),
                        [clf1, clf2, clf3, clf4, clf5],
                        ['Decision Tree (depth=10)', 'KNN (k=1)',
                         'Kernel SVM', 'Linear SVM', 'Naive Bayes']):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,
                                  s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt)

plt.show()

"""
+ Korzystając z [tego](http://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_decision_regions.html) przykładu wyświetl na jednym wykresie granice decyzyjne dla następujących klasyfikatorów:
 + KNN z liczbą najbliższych sąsiadów 1; x
 + Liniowy SVM; x
 + SVM z jądrem RBF; x
 + Naive Bayes; x
 + Drzewa dacyzyjnego o maksymalnej głębokosci 10. x
 
"""

