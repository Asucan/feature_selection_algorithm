# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:49:46 2019

@author: Asucan
"""

from __future__ import print_function
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import minmax_scale
from scipy.spatial.distance import euclidean


class ReliefF(object):

    def __init__(self,n_neighbours=1):
        """Sets up ReliefF to perform feature selection.

        Parameters
        ----------
        n_features_to_keep: int (default: 10)
            The number of top features (according to the ReliefF score) to retain after
            feature selection is applied.

        Returns
        -------
        None

        """

        self.feature_scores = None
        self.top_features = None
        self.n_neighbours = n_neighbours+1


    def fit(self, X, y, scaled=True):
        """Computes the feature importance scores from the training data.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels
        scaled: Boolen
            whether scale X ro not

        Returns
        -------
        self.top_features
        self.feature_scores

        """
        if scaled:
            X = minmax_scale( X, feature_range=(0, 1), axis=0, copy=True)

        self.feature_scores = np.zeros(X.shape[1], dtype=np.float64)

        # The number of labels and its corresponding prior probability
        labels, counts = np.unique(y, return_counts=True)
        Prob = counts/float(len(y))

        for label in labels:
            # Find the near-hit for each sample in the subset with label 'label'
            select = (y == label)
            tree = KDTree(X[select, :])
            nh = tree.query(X[select, :], k=self.n_neighbours, return_distance=False)[:, 1:].T
            #print(nh)
            nh_mat = np.zeros_like(X[select,:])
            for i in range(self.n_neighbours-1):
                t = (nh[i]).tolist()
                #print(len(nh))

                # calculate -diff(x, x_nh) for each feature of each sample 
                # in the subset with label 'label'
                #print(X[select, :].shape)
                #print(X[select, :][nh,:].shape)
                nh_mat += np.abs(np.subtract(X[select, :], X[select, :][t, :] ) ) * -1*(1/float(self.n_neighbours-1))

            # Find the near-miss for each sample in the other subset
            nm_mat = np.zeros_like(X[select, :])
            for prob, other_label in zip(Prob[labels != label], labels[labels != label] ):
                other_select = (y == other_label)
                other_tree = KDTree(X[other_select, :])
                nm = other_tree.query(X[select, :], k=self.n_neighbours-1, return_distance=False).T
                # calculate -diff(x, x_nm) for each feature of each sample in the subset 
                # with label 'other_label'
                nm_tmp = np.zeros_like(X[select, :])
                for i in range(self.n_neighbours-1):
                    t = nm[i].tolist()
                    nm_tmp += np.abs(np.subtract(X[select, :], X[other_select, :][t, :] ) ) * prob*(1/float(self.n_neighbours-1))
                nm_mat = np.add(nm_mat, nm_tmp)

            mat = np.add(nh_mat, nm_mat)
            self.feature_scores += np.sum(mat, axis=0)
            #print(self.feature_scores)

        # Compute indices of top features, cast scores to floating point.
        self.top_features = np.argsort(self.feature_scores)[::-1]
        self.feature_scores = self.feature_scores[self.top_features]

        return #self.top_features, self.feature_scores

    def transform(self, X, n_features_to_keep=10):
        """Reduces the feature set down to the top `n_features_to_keep` features.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Feature matrix to perform feature selection on

        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix

        """
        return X[:, self.top_features[:n_features_to_keep] ]

    def fit_transform(self, X, y,n_features_to_keep=10):
        """Computes the feature importance scores from the training data, then
        reduces the feature set down to the top `n_features_to_keep` features.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels

        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix

        """
        self.fit(X, y)
        return self.transform(X, n_features_to_keep)