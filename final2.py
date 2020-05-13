import csv
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, Model

output_folder_name = 'Output2'
data_folder_name = 'FraudedRawData'
label_file_name = 'challengeToFill.csv'

# Threshold of precentage above average distance from cluster
# in order to classify instance as postive (written by masq)
# 1.2 means that the maximum allowed distance above the average can be 20 precent.
UP_KMENAS_THRESH = 1.15

num_of_users = 40
num_of_labeled_users = 10
num_of_segments = 150
num_of_genuine_segments = 50
num_of_words_in_seg = 100

def read_data():

    user_data = []

    for user in range(0, num_of_users):

        user_file = os.path.abspath(os.path.join(data_folder_name, 'User' + str(user)))
        user_df = pd.read_csv(user_file, header=None, names=['Vocalbulary'])

        user_segs = []

        for seg in range(0, num_of_segments):
            curr_seg = seg * num_of_words_in_seg

            user_seg = user_df['Vocalbulary'][curr_seg:curr_seg + num_of_words_in_seg].tolist()

            user_segs.append(' '.join(user_seg))

        user_data.append(user_segs)

    return np.array(user_data)

def read_labels():

    label_file = os.path.abspath(os.path.join(label_file_name))
    label_file_df = pd.read_csv(label_file, index_col=0)

    user_labels = []

    for user in range(0, num_of_users):

        user_label = []

        for seg in range(0, num_of_segments):
            label = label_file_df.values[user, seg]
            user_label.append(label)

        user_labels.append(user_label)

    return np.array(user_labels)

def predict_by_euclidian_distance(X, clf):
    """Classify every instance in X based on his distance from the center
       of clf clust.
    """
    preds = []

    dists_from_cluster = euclidean_distances(
        X, clf.cluster_centers_)
    avg_dist_from_cluster = dists_from_cluster.mean()
    # For every instance check his distance relative to the avg
    # distance from the center of the cluster.
    for dist in dists_from_cluster:
        if dist >= avg_dist_from_cluster:
            if (dist / avg_dist_from_cluster) > UP_KMENAS_THRESH:
                preds.append(1)
            else:
                preds.append(0)
        elif dist < avg_dist_from_cluster:
            preds.append(0)
    return preds

def main():

    X = read_data()

    Y = read_labels()

    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    isf = IsolationForest()
    lof = LocalOutlierFactor(novelty=True)
    svm = OneClassSVM(kernel='rbf')
    cov = EllipticEnvelope()
    kmn = KMeans(n_clusters=1)
    gmm = GaussianMixture(n_components=2)

    preds_isf = []
    preds_lof = []
    preds_svm = []
    preds_cov =[]
    preds_kmn =[]
    preds_gmm = []

    preds= []

    for user in range(0,num_of_labeled_users):
        X_all = X[user]
        Y_all = Y[user]

        count_vect = CountVectorizer()
        tfidf_transformer = TfidfTransformer(use_idf=False)

        X_all_counts = count_vect.fit_transform(X_all)
        X_all_tfidf = tfidf_transformer.fit_transform(X_all_counts)

        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        max_features = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
        max_samples = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
        max_samples.append('auto')
        contamination = [0.001, 0.1, 0.2, 0.5]
        bootstrap = [True, False]

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_samples': max_samples,
                       'contamination': contamination,
                       'bootstrap': bootstrap}

        isf_random = RandomizedSearchCV(estimator=isf, param_distributions=random_grid, n_iter=1, cv=3, verbose=2,
                                       random_state=42, n_jobs=-1, scoring=make_scorer(accuracy_score))

        isf_random.fit(X_all_tfidf, Y_all)


        nu = [0.05,0.1,0.15,0.2]
        gamma = [0.01,0.05,0.1,0.2]
        kernel = ['rbf']


        # Create the random grid
        random_grid = {'nu': nu,
                       'kernel': kernel,
                       'gamma': gamma}

        svm_random = RandomizedSearchCV(estimator=svm, param_distributions=random_grid, n_iter=50, cv=3, verbose=2,
                                        random_state=42, n_jobs=-1, scoring=make_scorer(accuracy_score))

        svm_random.fit(X_all_tfidf, Y_all)

    for user in range(num_of_labeled_users,num_of_users):
        X_all = X[user]
        X_labeled = X[user][0:num_of_genuine_segments]
        X_unlabeled = X[user][num_of_genuine_segments:]

        Y_all = Y[user]
        Y_labeled = Y[user][0:num_of_genuine_segments]
        Y_unlabeled = Y[user][num_of_genuine_segments:]

        count_vect = CountVectorizer()
        tfidf_transformer = TfidfTransformer(use_idf=False)

        X_all_counts = count_vect.fit_transform(X_all)
        X_labeled_counts = count_vect.transform(X_labeled)
        X_unlabeled_counts = count_vect.transform(X_unlabeled)

        X_all_tfidf = tfidf_transformer.fit_transform(X_all_counts)
        X_labeled_tfidf = tfidf_transformer.transform(X_labeled_counts)
        X_unlabeled_tfidf = tfidf_transformer.transform(X_unlabeled_counts)

        isf.fit(X_all_tfidf)
        lof.fit(X_all_tfidf)
        svm.fit(X_all_tfidf)
        cov.fit(X_all_tfidf.toarray())
        kmn.fit(X_all_tfidf)
        gmm.fit(X_all_tfidf.toarray())

        pred_isf = isf.predict(X_unlabeled_tfidf)
        pred_lof = lof.predict(X_unlabeled_tfidf)
        pred_svm = svm.predict(X_unlabeled_tfidf)
        pred_cov = cov.predict(X_unlabeled_tfidf.toarray())
        pred_kmn = predict_by_euclidian_distance(X_unlabeled_tfidf.toarray(), kmn)
        pred_gmm = gmm.predict(X_unlabeled_tfidf.toarray())

        pred_isf = [1 if p == -1 else 0 for p in pred_isf]
        pred_lof = [1 if p == -1 else 0 for p in pred_lof]
        pred_svm = [1 if p == -1 else 0 for p in pred_svm]
        pred_cov = [1 if p == -1 else 0 for p in pred_cov]
        pred_kmn = [1 if p == -1 else 0 for p in pred_kmn]
        pred_gmm = [1 if p == -1 else 0 for p in pred_gmm]

        preds_lof.append(pred_lof)
        preds_isf.append(pred_isf)
        preds_svm.append(pred_svm)
        preds_cov.append(pred_cov)
        preds_kmn.append(pred_kmn)
        preds_gmm.append(pred_gmm)

        pred_sum = np.array(pred_lof) + np.array(pred_isf) + np.array(pred_svm) +\
                   np.array(pred_cov) + np.array(pred_kmn) + np.array(pred_gmm)

        majority = [1 if i > 3 else 0 for i in pred_sum]
        preds.append(majority)

if __name__ == '__main__':
    main()

