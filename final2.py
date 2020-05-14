import os

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

def main():

    X = read_data()

    Y = read_labels()

    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    isf = IsolationForest(contamination=0.24, max_samples=100, n_estimators=100)
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, leaf_size=30, algorithm='auto', contamination=0.3)
    svm = OneClassSVM(nu=0.05, kernel="rbf", gamma=0.01)
    cov = EllipticEnvelope(contamination=0.01,support_fraction=0.94)
    kmn = KMeans(n_clusters=1, n_init=10)
    gmm = GaussianMixture(n_components=1)
    dbs = DBSCAN(eps=.2, metric='euclidean', min_samples=5, n_jobs=-1)

    '''
    params_isf = []
    params_lof = []
    params_svm = []
    params_cov = []
    params_kmn = []
    params_gmm = []
    params_dbs = []

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

        isf_random = RandomizedSearchCV(estimator=isf, param_distributions=random_grid, n_iter=10, cv=3, verbose=2,
                                       random_state=42, n_jobs=-1, scoring=make_scorer(accuracy_score))

        isf_random.fit(X_all_tfidf, Y_all)

        params_isf.append(isf_random.best_params_)

        nu = [0.05,0.1,0.15,0.2]
        gamma = [0.01,0.05,0.1,0.2]
        kernel = ['rbf']


        # Create the random grid
        random_grid = {'nu': nu,
                       'kernel': kernel,
                       'gamma': gamma}

        svm_random = RandomizedSearchCV(estimator=svm, param_distributions=random_grid, n_iter=10, cv=3, verbose=2,
                                        random_state=42, n_jobs=-1, scoring=make_scorer(accuracy_score))

        svm_random.fit(X_all_tfidf, Y_all)

        params_svm.append(svm_random.best_params_)
'''

    preds_isf = []
    preds_lof = []
    preds_svm = []
    preds_cov = []
    preds_kmn = []
    preds_gmm = []
    preds_dbs = []

    preds = []

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


        stats_transformer = TextStats()
        DictVector = DictVectorizer()

        X_all_stats = stats_transformer.fit_transform(X_all)
        X_labeled_stats = stats_transformer.transform(X_labeled)
        X_unlabeled_stats = stats_transformer.transform(X_unlabeled)

        X_all_vect = DictVector.fit_transform(X_all_stats)
        X_labeled_vect = DictVector.transform(X_labeled_stats)
        X_unlabeled_vect = DictVector.transform(X_unlabeled_stats)

        fu = FeatureUnion(
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
                transformer_weights={
                    'body_bow': 0.5,
                    'body_stats': 1.0,
                })

        X_all_tfidf=fu.fit_transform(X_all)
        X_labeled_tfidf = fu.transform(X_labeled)
        X_unlabeled_tfidf = fu.transform(X_unlabeled)

        '''w2v_model = Word2Vec(size=100, window=5, min_count=1, workers=4)
        w2v_model.build_vocab(X_all, progress_per=10000)
        w2v_model.train(X_all, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
        w2v_model.init_sims(replace=True)'''

        isf.fit(X_all_tfidf)
        lof.fit(X_all_tfidf)
        svm.fit(X_all_tfidf)
        cov.fit(X_all_tfidf.toarray())
        kmn.fit(X_all_tfidf)
        gmm.fit(X_all_tfidf.toarray())
        pred_dbs = dbs.fit_predict(X_unlabeled_tfidf)

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
        pred_gmm = [1 if p == -1 else 0 for p in pred_gmm]
        pred_dbs = [1 if p == -1 else 0 for p in pred_dbs]

        preds_lof.append(pred_lof)
        preds_isf.append(pred_isf)
        preds_svm.append(pred_svm)
        preds_cov.append(pred_cov)
        preds_kmn.append(pred_kmn)
        preds_gmm.append(pred_gmm)
        preds_dbs.append(pred_dbs)

        pred_sum = np.array(pred_lof) + np.array(pred_isf) + np.array(pred_svm) +np.array(pred_kmn) +\
                   np.array(pred_cov) +  + np.array(pred_gmm) + np.array(pred_dbs)

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
    print("KMEANS:")
    evaluate_model(preds_kmn, Y)
    print("GMM:")
    evaluate_model(preds_gmm, Y)
    print("DBS:")
    evaluate_model(preds_dbs, Y)

    print("TOTAL:")
    evaluate_model(preds, Y)

if __name__ == '__main__':
    main()

