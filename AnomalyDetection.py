import os
import shutil

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.pipeline import Pipeline

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion


output_folder_name = 'Output'
data_folder_name = 'FraudedRawData'
label_file_name = 'challengeToFill.csv'

num_of_users = 40
num_of_labeled_users = 10
num_of_segments = 150
num_of_genuine_segments = 50
num_of_words_in_seg = 100


ISF_HYPER_PARAMS = {
    "contamination": 0.25,
    "max_samples": 100,
    "max_features": 60,
    "n_estimators": 100
}

LOF_HYPER_PARAMS = {
    "n_neighbors": 40,
    "novelty": True,
    "leaf_size": 70,
    "algorithm": 'auto',
    "contamination": 0.25,
    "metric": 'euclidean'
}

SVM_HYPER_PARAMS = {
    "nu": 0.3,
    "kernel": "rbf",
    "gamma": 0.4
}

COV_HYPER_PARAMS = {
    "contamination": 0.1,
    "support_fraction": 0.5
}

KMN_HYPER_PARAMS = {
    "n_clusters": 1,
    "n_init": 10,
    "max_iter": 500
}

KMENAS_THRESH = 1.15

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

def evaluate_model(preds, labels):
    """Evaluate the model described by the given predictions.

    The evaluation is based on comparing the first 10 users to the true
    predictions available in the partial_labels.csv file.
    """
    ACC, TN, FN, TP, FP = 0, 0, 0, 0, 0

    for user in range(num_of_labeled_users):
        user_preds = preds[user]
        user_labels = labels[user][num_of_genuine_segments:]
        ACC += accuracy_score(user_labels, user_preds)
        cm = confusion_matrix(user_labels, user_preds)
        TN += cm[0][0]
        FN += cm[1][0]
        TP += cm[1][1]
        FP += cm[0][1]

    print("#" * 10)
    print("average accuracy = %s" % (ACC / num_of_labeled_users))
    print("average true_negative = %s" % (TN / num_of_labeled_users))
    print("average false_negative = %s" % (FN / num_of_labeled_users))
    print("average true_positive = %s" % (TP / num_of_labeled_users))
    print("average false_positive = %s" % (FP / num_of_labeled_users))
    print("#" * 10)

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
            if (dist / avg_dist_from_cluster) > KMENAS_THRESH:
                preds.append(1)
            else:
                preds.append(0)
        elif dist < avg_dist_from_cluster:
            preds.append(0)
    return preds

def WriteOutput(y_pred):

    input_file = os.path.abspath(os.path.join(label_file_name))
    input_file_df = pd.read_csv(input_file, index_col=0)

    output_df = input_file_df.values

    output_df[num_of_labeled_users:num_of_users, num_of_genuine_segments:num_of_segments] = y_pred[num_of_labeled_users:num_of_users]

    output_df = pd.DataFrame(data=output_df, index=input_file_df.index, columns=input_file_df.columns, dtype=int)

    output_file = os.path.abspath(os.path.join(output_folder_name, label_file_name))

    if os.path.exists(os.path.abspath(output_folder_name)):
        shutil.rmtree(os.path.abspath(output_folder_name))
    os.makedirs(os.path.abspath(output_folder_name))

    output_df.to_csv(output_file, index=True)


def main():

    X = read_data()

    Y = read_labels()

    isf = IsolationForest(**ISF_HYPER_PARAMS)
    lof = LocalOutlierFactor(**LOF_HYPER_PARAMS)
    svm = OneClassSVM(**SVM_HYPER_PARAMS)
    cov = EllipticEnvelope(**COV_HYPER_PARAMS)
    kmn = KMeans(**KMN_HYPER_PARAMS)

    preds_isf = []
    preds_lof = []
    preds_svm = []
    preds_cov = []
    preds_kmn = []

    preds = []

    for user in range(0,num_of_users):
        X_all = X[user]
        X_labeled = X[user][0:num_of_genuine_segments]
        X_unlabeled = X[user][num_of_genuine_segments:]

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

        pred_isf = isf.predict(X_unlabeled_tfidf)
        pred_lof = lof.predict(X_unlabeled_tfidf)
        pred_svm = svm.predict(X_unlabeled_tfidf)
        pred_cov = cov.predict(X_unlabeled_tfidf.toarray())
        pred_kmn = predict_by_euclidian_distance(X_unlabeled_tfidf, kmn)

        pred_isf = [1 if p == -1 else 0 for p in pred_isf]
        pred_lof = [1 if p == -1 else 0 for p in pred_lof]
        pred_svm = [1 if p == -1 else 0 for p in pred_svm]
        pred_cov = [1 if p == -1 else 0 for p in pred_cov]

        preds_lof.append(pred_lof)
        preds_isf.append(pred_isf)
        preds_svm.append(pred_svm)
        preds_cov.append(pred_cov)
        preds_kmn.append(pred_kmn)

        pred_sum = np.array(pred_lof) + np.array(pred_isf) + np.array(pred_svm) + np.array(pred_kmn)  + np.array(pred_cov)

        majority = [1 if i > 2 else 0 for i in pred_sum]
        preds.append(majority)


    print("LOF:")
    evaluate_model(preds_lof, Y)
    print("ISF:")
    evaluate_model(preds_isf, Y)
    print("SVM:")
    evaluate_model(preds_svm, Y)
    print("COV:")
    evaluate_model(preds_cov, Y)
    print("KMN:")
    evaluate_model(preds_kmn, Y)

    print("TOTAL:")
    evaluate_model(preds, Y)

    WriteOutput(preds)

if __name__ == '__main__':
    main()

