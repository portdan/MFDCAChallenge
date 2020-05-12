import numpy as np
import pandas as pd
import os.path

from IPython import embed

# tfidf
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# classifiers
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.naive_bayes import MultinomialNB

DATASET_PATH = "FraudedRawData"
SINGLE_DATASET_FILE = os.path.join(DATASET_PATH, "User")

PARTIAL_LABELS = "partial_labels.csv"

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

CLASSIFIER_HYPER_PARAMS = {
    "behaviour": 'new',
    "contamination": 0.32,
    "max_samples": 100,
    "n_estimators": 200
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
        clf = IsolationForest(**CLASSIFIER_HYPER_PARAMS)

        X_train = users_segs[user]
        X_test = users_segs[user][NUM_GENUINE_SEGS:]

        ##################
        #  MODEL FITING  #
        ##################

        pipe_clf = Pipeline([('vect', count_vect),
                             ('tfidf', tfidf_transformer),
                             ('clf', clf),
                             ])
        pipe_clf.fit(X_train)

        ##################
        #   PREDICTING   #
        ##################

        pred_segs = pipe_clf.predict(X_test)
        # Isolation forest returns -1 for abnormal and 1 for normal behaviours.
        # replace -1 with 1 and 1 with 0
        preds[user] = [1 if p == -1 else 0 for p in pred_segs]

        # TODO: use GridSearchCV to optimate hyper parameters.
    evaluate_model(preds, segs_partial_labels)
    create_submission(preds)


if __name__ == '__main__':
    main()
