import numpy as np
import pandas as pd
import os.path

from IPython import embed

# k-means
from sklearn.neighbors import KernelDensity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# classifiers
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import silhouette_score
from sklearn.neighbors import LocalOutlierFactor

DATASET_PATH = "FraudedRawData"
SINGLE_DATASET_FILE = os.path.join(DATASET_PATH, "User")

PARTIAL_LABELS = "partial_labels.csv"

NUM_USERS = 40

SEGMENT_SIZE = 100
NUM_SEGMENTS = 150
NUM_GENUINE_SEGS = 50

# Threshold of precentage above average distance from cluster
# in order to classify instance as postive (written by masq)
# 1.2 means that the maximum allowed distance above the average can be 20 precent.
UP_KMENAS_THRESH = 1.3
# In order recognize better "good" segments, our KMeans model will have a changing
# number of clusters - according to the silhoutte factor.
SILHOUETTE_THRES = 0.2
SILHOUETTE_N_CLUSTERS = [2, 3, 4, 5]

VECTORIZER_HYPER_PARAMS = {
    "max_features": 150,
    "ngram_range": (1, 1),
}

TRANSFORMER_HYPER_PARAMS = {
    "use_idf": False
}

CLASSIFIER_HYPER_PARAMS = {
    "n_clusters": 1,
    "n_init": 10
}

###############
#    UTILS    #
###############


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
    df.drop(columns=["id"], axis=1, inplace=True)
    return df.T


def create_clf_with_silo(X):
    sf = []
    n_clusters = 1
    for clusters in SILHOUETTE_N_CLUSTERS:
        hp = CLASSIFIER_HYPER_PARAMS
        hp["n_clusters"] = clusters
        clf = KMeans(**hp).fit(X)
        sf.append(silhouette_score(X, clf.labels_))
    max_sf = max(sf)
    if max_sf >= SILHOUETTE_THRES:
        n_clusters = SILHOUETTE_N_CLUSTERS[sf.index(max_sf)]
    hp["n_clusters"] = n_clusters
    return KMeans(**hp).fit(X)


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


def predict_by_euclidian_distance_clusters(X, clf):
    """Classify every instance in X based on his distance from the center
       of clf clust.
    """
    preds = []

    for seg in X:
        clusters_dist = list(euclidean_distances(
            [seg], clf.cluster_centers_))[0]
        clusters_dist.sort()
        # take 2 minimal distances
        min_dists = clusters_dist[:2]
        print(min_dists)
        if (min_dists[1] / min_dists[0]) < UP_KMENAS_THRESH:
            preds.append(1)
        else:
            preds.append(0)

    return preds


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
        print("="*10 + "User" + str(user) + "="*10)
        user_preds = preds[user]
        user_true_preds = list(true_preds[user][NUM_GENUINE_SEGS:])
        accuracy = accuracy_score(user_true_preds, user_preds)
        print("accuracy = %s" % accuracy)
        ttl_acc += accuracy

        z = zip(user_true_preds, user_preds)
        true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0

        for true, pred in z:
            if pred:
                if pred != true:
                    false_positive += 1
                else:
                    true_positive += 1
            elif not pred:
                if pred != true:
                    false_negative += 1
                else:
                    true_negative += 1

        print("true_negative=%s (out of %s)" %
              (str(true_negative), user_true_preds.count(0)))
        ttl_tn += true_negative
        print("false_negative=" + str(false_negative))
        ttl_fn += false_negative
        print("true_positive=%s (out of %s)" %
              (str(true_positive), user_true_preds.count(1)))
        ttl_tp += true_positive
        print("false_positive=" + str(false_positive))
        ttl_fp += false_positive
        print(list(zip(user_true_preds, user_preds)))
        print("=" * 25)

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
    users_segs = load_users_segs_str()
    segs_partial_labels = load_partial_labels()

    for user in users_segs.keys():
        count_vect = CountVectorizer(**VECTORIZER_HYPER_PARAMS)
        tfidf_transformer = TfidfTransformer(**TRANSFORMER_HYPER_PARAMS)
        clf = KMeans(**CLASSIFIER_HYPER_PARAMS)

        X_train = users_segs[user][:NUM_GENUINE_SEGS]
        X_test = users_segs[user][NUM_GENUINE_SEGS:]

        ##################
        #  MODEL FITING  #
        ##################

        count_vect = CountVectorizer(**VECTORIZER_HYPER_PARAMS)
        X_train_counts = count_vect.fit_transform(X_train)
        tfidf_transformer = TfidfTransformer(**TRANSFORMER_HYPER_PARAMS)
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        X_test_counts = count_vect.transform(X_test)
        X_test_tfidf = tfidf_transformer.transform(X_test_counts).toarray()
        clf = create_clf_with_silo(X_train_tfidf)

        clf.fit(X_train_tfidf)
        preds[user] = predict_by_euclidian_distance_clusters(X_test_tfidf, clf)

        ##################
        #   PREDICTING   #
        ##################

    evaluate_model(preds, segs_partial_labels)
    create_submission(preds)


if __name__ == '__main__':
    main()
