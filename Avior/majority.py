import string

import numpy as np
import pandas as pd
import os.path

#from IPython import embed

# tfidf
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.covariance import EllipticEnvelope

# k-means
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from sklearn.metrics.pairwise import euclidean_distances

# Threshold of precentage above average distance from cluster
# in order to classify instance as postive (written by masq)
# 1.2 means that the maximum allowed distance above the average can be 20 precent.
UP_KMENAS_THRESH = 1.15
# In order recognize better "good" segments, our KMeans model will have a changing
# number of clusters - according to the silhoutte factor.
SILHOUETTE_THRES = 0.
SILHOUETTE_N_CLUSTERS = [2, 3, 4]


# features
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import FeatureUnion
from collections import Counter


# classifiers
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import OneClassSVM

DATASET_PATH = "FraudedRawData"
SINGLE_DATASET_FILE = os.path.join(DATASET_PATH, "User")

PARTIAL_LABELS = "challengeToFill.csv"

NUM_USERS = 40

SEGMENT_SIZE = 100
NUM_SEGMENTS = 150
NUM_GENUINE_SEGS = 50

VECTORIZER_HYPER_PARAMS = {
    "max_features": 150,
    "ngram_range": (1, 1),
    # "max_df": 0.95
}

TRANSFORMER_HYPER_PARAMS = {
    "use_idf": False
}

KMEANS_HYPER_PARAMS = {
    "n_clusters": 1,
    "n_init": 10
}

ISF_HYPER_PARAMS = {
    "behaviour": 'new',
    "contamination": 0.24,  # change
    "max_samples": 150,  # change
    "n_estimators": 100  #change
}

LOF_HYPER_PARAMS = {
    "n_neighbors": 20,
    "novelty": True,
    "leaf_size": 30,
    "algorithm": 'auto',
    "contamination": 0.3
}

SVM_HYPER_PARAMS = {
    "nu": 0.05,
    "kernel": "rbf",
    "gamma": 0.01
}

COV_HYPER_PARAMS = {
    "contamination": 0.01,
    "support_fraction": 0.94
}

###############
#    UTILS    #
###############


class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""
    def fit(self, x, y=None):
        return self

    def transform(self, usr):
        return [{'length': len(seg),
                 'num_Upper_case': sum(1 for c in seg if c.isupper())}
               #  'average_num_command': avg_frequent_cmd(seg),
              #   'count_max_command': count_most_frequent_cmd(seg)}
                for seg in usr]

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

def count_most_frequent_cmd(seg, num=1):
    # split() returns list of all the words in the string
    split_it = seg.split()
    # Pass the split_it list to instance of Counter class.
    count = Counter(split_it)
    # most_common() produces k frequently encountered
    # input values and their respective counts.
    most_occur = count.most_common(num)
    return most_occur[0][1]


def avg_frequent_cmd(seg):
    # split() returns list of all the words in the string
    split_it = seg.split()
    # Pass the split_it list to instance of Counter class.
    count = Counter(split_it)
    lst = [item for item in count.values()]
    return sum(lst) / len(lst)


def load_users_segs_str():
    users_to_segs_dict = {}
    for count in range(0, NUM_USERS):
        with open(SINGLE_DATASET_FILE + str(count), 'r') as file:
            lines = list(map(lambda s: s.strip(), file.readlines()))
            segs = []
            for seg in range(NUM_SEGMENTS):
                curr_seg = seg * SEGMENT_SIZE
                segs.append(' '.join(lines[curr_seg:curr_seg + SEGMENT_SIZE]))
            users_to_segs_dict[count] = segs
    return users_to_segs_dict


def load_users_segs_lst():
    """convert users file to dict(file/user) of lists(segment) of elements(command)"""
    users_to_segs_dict = {}
    for count in range(0, NUM_USERS):
        with open(SINGLE_DATASET_FILE + str(count), 'r') as file:
            lines = list(map(lambda s: s.strip(), file.readlines()))
            segs = []
            for seg in range(NUM_SEGMENTS):
                curr_seg = seg * SEGMENT_SIZE
                segs.append(lines[curr_seg:curr_seg + SEGMENT_SIZE])
            users_to_segs_dict[count] = segs
    return users_to_segs_dict


def load_partial_labels():
    """Returns DataFrame in which each column represent user."""
    df = pd.read_csv(PARTIAL_LABELS)
    df.drop(df.columns[0], axis=1, inplace=True)
    return df.T


def convert_to_flat_list(lst):
    """
    flat_list = []
    for sublist in l:
        for item in sublist:
            flat_list.append(item)"""
    return [item for sublist in lst for item in sublist]


def only_tf(X_train_counts):
    # need to check if that realy just tf
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    print(X_train_tf.shape)
    return X_train_tf


def tfidf(X_counts):
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    print(X_tfidf.shape)
    return X_tfidf


def check_new_object(count_vect, new_seg, clf):
    X_new_counts = count_vect.transform(new_seg)
    X_new_tfidf = tfidf(X_new_counts)
    predicted = clf.predict(X_new_tfidf)
    for doc, category in zip(new_seg, predicted):
        pass


def create_submission(preds):
    df = pd.DataFrame()
    for user in range(10, NUM_USERS):
        user_preds = preds[user]
        for idx, pred in enumerate(user_preds):
            start_seg = NUM_GENUINE_SEGS * SEGMENT_SIZE + idx * SEGMENT_SIZE
            end_seg = start_seg + SEGMENT_SIZE
            idstr = "User{}_{}-{}".format(user, start_seg, end_seg)
            df = df.append(pd.Series([idstr, pred]), ignore_index=True)
    df.columns = ["id", "label"]
    df.label = df.label.astype(int)
    df.to_csv("submission.csv", index=False)


def evaluate_model(preds, true_preds):
    """Evaluate the model described by the given predictions.

    The evaluation is based on comparing the first 10 users to the true
    predictions available in the partial_labels.csv file.
    """
    ttl_acc, ttl_fp, ttl_tp, ttl_fn, ttl_tn = 0, 0, 0, 0, 0
    for user in range(10):
    #    print("="*10 + "User" + str(user) + "="*10)
        user_preds = preds[user]
        user_true_preds = list(true_preds[user][NUM_GENUINE_SEGS:])
        accuracy = accuracy_score(user_true_preds, user_preds)
     #   print("accuracy = %s" % accuracy)
        ttl_acc += accuracy

        z = zip(user_true_preds, user_preds)
        true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0

        for true, pred in z:
            if pred:
                if pred != true: # pred true real false
                    false_positive += 1
                else:
                    true_positive += 1
            elif not pred:
                if pred != true:
                    false_negative += 1
                else:
                    true_negative += 1

    #    print("true_negative=%s (out of %s)" %
    #          (str(true_negative), user_true_preds.count(0)))
        ttl_tn += true_negative
    #    print("false_negative=" + str(false_negative))
        ttl_fn += false_negative
    #    print("true_positive=%s (out of %s)" %
     #         (str(true_positive), user_true_preds.count(1)))
        ttl_tp += true_positive
    #    print("false_positive=" + str(false_positive))
        ttl_fp += false_positive
    #    print(list(zip(user_true_preds, user_preds)))
    #    print("=" * 25)

    print("#" * 25)
    print("avg accuracy=%s" % (ttl_acc / 10.0))
    print("avg true_negative=%s" % (ttl_tn / 10.0))
    print("avg false_negative=%s" % (ttl_fn / 10.0))
    print("avg true_positive=%s" % (ttl_tp / 10.0))
    print("avg false_positive=%s" % (ttl_fp / 10.0))
    print("#" * 25)


def main():
    # Load users segments from given dataset.
    # Each segment is loaded as a concatenated string of the contained lines.
    preds = {}
    preds_kmeans = {}
    preds_isf = {}
    preds_lof = {}
    preds_svm = {}
    users_segs = load_users_segs_str()
    segs_partial_labels = load_partial_labels()

    count_vect = CountVectorizer(**VECTORIZER_HYPER_PARAMS)
    tfidf_transformer = TfidfTransformer(**TRANSFORMER_HYPER_PARAMS)
    stats_transformer = TextStats()
    DictVector = DictVectorizer()
    kmeans = KMeans(**KMEANS_HYPER_PARAMS)
    isf = IsolationForest(**ISF_HYPER_PARAMS)
    lof = LocalOutlierFactor(**LOF_HYPER_PARAMS)
    svm = OneClassSVM(**SVM_HYPER_PARAMS)
    cov = EllipticEnvelope(**COV_HYPER_PARAMS)

    for user in users_segs.keys():
        X_train = users_segs[user]
        X_test = users_segs[user][NUM_GENUINE_SEGS:]
        X_train_lite = users_segs[user][:NUM_GENUINE_SEGS]

        ##################
        #  MODEL FITING  #
        ##################

        # pipe_kmeans = Pipeline([
        #     # Use FeatureUnion to combine the features from subject and body
        #     ('union', FeatureUnion(
        #         transformer_list=[
        #
        #             # Pipeline for standard bag-of-words model for body
        #             ('body_bow', Pipeline([
        #                 ('vect', count_vect),
        #                 ('tfidf', tfidf_transformer),
        #             ])),
        #
        #             # Pipeline for pulling ad hoc features from post's body
        #             ('body_stats', Pipeline([
        #                 ('stats', stats_transformer),  # returns a list of dicts
        #                 ('vect', DictVector),  # list of dicts -> feature matrix
        #             ])),
        #         ],
        #
        #         # weight components in FeatureUnion
        #         transformer_weights={
        #             'body_bow': 0.5,
        #             'body_stats': 0,
        #         },
        #     )),
        #
        #     ('kmeans', kmeans),
        # ])

        pipe_lof = Pipeline([
            # Use FeatureUnion to combine the features from subject and body
            ('union', FeatureUnion(
                transformer_list=[

                    # Pipeline for standard bag-of-words model for body
                    ('body_bow', Pipeline([
                        ('vect', count_vect),
                        ('tfidf', tfidf_transformer),
                    ])),

                    # Pipeline for pulling ad hoc features from post's body
                    ('body_stats', Pipeline([
                        ('stats', stats_transformer),  # returns a list of dicts
                        ('vect', DictVector),  # list of dicts -> feature matrix
                    ])),
                ],

                # weight components in FeatureUnion
                transformer_weights={
                    'body_bow': 0.5,
                    'body_stats': 0.5,
                },
            )),

            ('lof', lof),
        ])

        pipe_isf = Pipeline([
            # Use FeatureUnion to combine the features from subject and body
            ('union', FeatureUnion(
                transformer_list=[

                    # Pipeline for standard bag-of-words model for body
                    ('body_bow', Pipeline([
                        ('vect', count_vect),
                        ('tfidf', tfidf_transformer),
                    ])),

                    # Pipeline for pulling ad hoc features from post's body
                    ('body_stats', Pipeline([
                        ('stats', stats_transformer),  # returns a list of dicts
                        ('vect', DictVector),  # list of dicts -> feature matrix
                    ])),
                ],
                # weight components in FeatureUnion
                transformer_weights={
                    'body_bow': 0.5,
                    'body_stats': 1.0,
                },
            )),
            ('isf', isf),
        ])


        pipe_svm = Pipeline([
            # Use FeatureUnion to combine the features from subject and body
            ('union', FeatureUnion(
                transformer_list=[

                    # Pipeline for standard bag-of-words model for body
                    ('body_bow', Pipeline([
                        ('vect', count_vect),
                        ('tfidf', tfidf_transformer),
                    ])),

                    # Pipeline for pulling ad hoc features from post's body
                    ('body_stats', Pipeline([
                        ('stats', stats_transformer),  # returns a list of dicts
                        ('vect', DictVector),  # list of dicts -> feature matrix
                    ])),

                ],

                # weight components in FeatureUnion
                transformer_weights={
                    'body_bow': 0.5,
                    'body_stats': 1.0,
                },
            )),
            ('svm', svm),
        ])

        pipe_svm = Pipeline([
            # Use FeatureUnion to combine the features from subject and body
            ('union', FeatureUnion(
                transformer_list=[

                    # Pipeline for standard bag-of-words model for body
                    ('body_bow', Pipeline([
                        ('vect', count_vect),
                        ('tfidf', tfidf_transformer),
                    ])),

                    # Pipeline for pulling ad hoc features from post's body
                    ('body_stats', Pipeline([
                        ('stats', stats_transformer),  # returns a list of dicts
                        ('vect', DictVector),  # list of dicts -> feature matrix
                    ])),

                ],

                # weight components in FeatureUnion
                transformer_weights={
                    'body_bow': 0.5,
                    'body_stats': 1.0,
                },
            )),
            ('svm', svm),
        ])

        pipe_cov = Pipeline([
            # Use FeatureUnion to combine the features from subject and body
            ('union', FeatureUnion(
                transformer_list=[

                    # Pipeline for standard bag-of-words model for body
                    ('body_bow', Pipeline([
                        ('vect', count_vect),
                        ('tfidf', tfidf_transformer),
                    ])),

                    # Pipeline for pulling ad hoc features from post's body
                    ('body_stats', Pipeline([
                        # returns a list of dicts
                        ('stats', stats_transformer),
                        # list of dicts -> feature matrix
                        ('vect', DictVector),
                    ])),

                ],

                # weight components in FeatureUnion
                transformer_weights={
                    'body_bow': 0.5,
                    'body_stats': 1.0,
                },
            )),
            ('cov', cov),
        ])

        pipe_lof.fit(X_train)
        pipe_isf.fit(X_train)
#        pipe_cov.fit(X_train)


        ##################
        #   PREDICTING   #
        ##################

        pred_segs_lof = pipe_lof.predict(X_test)
        pred_segs_isf = pipe_isf.predict(X_test)

        pipe_svm.fit(X_train_lite)
        pred_segs_svm = pipe_svm.predict(X_test)
 #       pred_segs_cov = pipe_cov.predict(X_test)



        # Isolation forest returns -1 for abnormal and 1 for normal behaviours.
        # replace -1 with 1 and 1 with 0


        X_train_counts = count_vect.fit_transform(X_train)
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        X_test_counts = count_vect.transform(X_test)
        X_test_tfidf = tfidf_transformer.transform(X_test_counts).toarray()

        kmeans.fit(X_train_tfidf)

        preds_lof[user] = [1 if p == -1 else 0 for p in pred_segs_lof]
        preds_isf[user] = [1 if p == -1 else 0 for p in pred_segs_isf]
        preds_svm[user] = [1 if p == -1 else 0 for p in pred_segs_svm]
 #       preds_cov[user] = [1 if p == -1 else 0 for p in pred_segs_cov]
        preds_kmeans[user] = predict_by_euclidian_distance(X_test_tfidf, kmeans)

        temp = np.array(preds_lof[user]) + np.array(preds_isf[user]) + np.array(preds_svm[user]) +np.array(preds_kmeans[user])
        preds[user] = [1 if i > 2 else 0 for i in temp]

        # TODO: use GridSearchCV to optimate hyper parameters.
#    evaluate_model(preds, segs_partial_labels)
    print("KMEANS:")
    evaluate_model(preds_kmeans, segs_partial_labels)
    print("LOF:")
    evaluate_model(preds_lof, segs_partial_labels)
    print("isf:")
    evaluate_model(preds_isf, segs_partial_labels)
    print("SVM:")
    evaluate_model(preds_svm, segs_partial_labels)
    evaluate_model(preds, segs_partial_labels)
    create_submission(preds)
  #  create_submission(preds_un)


if __name__ == '__main__':
    main()
