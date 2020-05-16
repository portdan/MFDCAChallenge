import os
import shutil

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

output_folder_name = 'HyperParameterSearch'
data_folder_name = 'FraudedRawData'
label_file_name = 'challengeToFill.csv'

num_of_users = 40
num_of_labeled_users = 10
num_of_segments = 150
num_of_genuine_segments = 50
num_of_words_in_seg = 100

random_search_iter = 100

ISF_HYPER_PARAMS = {
    "n_estimators": [int(x) for x in np.linspace(start=5, stop=300, num=50)],
    "max_features": [int(x) for x in np.linspace(start=5, stop=150, num=25)],
    "max_samples": [int(x) for x in np.linspace(start=5, stop=100, num=20)],
    "contamination": [x for x in np.linspace(start=0.001, stop=0.5, num=25)],
    "bootstrap": [True, False]
}

LOF_HYPER_PARAMS = {
    "n_neighbors": [int(x) for x in np.linspace(start=10, stop=150, num=25)],
    "leaf_size": [int(x) for x in np.linspace(start=10, stop=150, num=25)],
    "algorithm": ['kd_tree'],
    "contamination": [x for x in np.linspace(start=0.001, stop=0.5, num=25)],
    "metric": ['euclidean']
}

SVM_HYPER_PARAMS = {
    "nu": [x for x in np.linspace(start=0.001, stop=1.0, num=50)],
    "gamma": [x for x in np.linspace(start=0.001, stop=1.0, num=50)],
    "kernel": ['rbf'],
    "degree": [int(x) for x in np.linspace(start=2, stop=10, num=8)]
}

KMN_HYPER_PARAMS = {
    "n_init": [int(x) for x in np.linspace(start=10, stop=100, num=25)],
    "init": ['k-means++', 'random'],
    "max_iter": [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
}

COV_HYPER_PARAMS = {
    "contamination": [x for x in np.linspace(start=0.001, stop=0.5, num=25)],
    "support_fraction": [x for x in np.linspace(start=0.001, stop=0.99, num=25)],
    "assume_centered" : [True, False]
}

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

def write_output(params, alg_name):

    df = pd.DataFrame(params)
    df.loc['mean'] = df.mean()
    df.loc['median'] = df.median()
    df.loc['std'] = df.std()

    alg_folder = os.path.abspath(os.path.join(output_folder_name,alg_name))
    alg_file = os.path.abspath(os.path.join(alg_folder, alg_name + ".csv"))

    if os.path.exists(alg_folder):
        shutil.rmtree(alg_folder)
    os.makedirs(alg_folder)

    df.to_csv(alg_file)

def custom_acc(y_true, y_pred):
    y_pred = [1 if p == -1 else 0 for p in y_pred]

    cm = confusion_matrix(y_true, y_pred)
    grade = 0

    if cm.shape == (1, 1):
        grade = np.sum(y_pred == y_true)/len(y_true)
    else:
        TN = cm[0][0]
        FN = cm[1][0]
        TP = cm[1][1]
        FP = cm[0][1]

        grade = (TN + TP*9) / ((FP+TN) + (TP+FN)*9)

    return grade


def calc_best_detector_for_algoritm(X, Y, clf, clf_params, k_fold):

    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer(use_idf=True)

    X_all_counts = count_vect.fit_transform(X)
    X_all_tfidf = tfidf_transformer.fit_transform(X_all_counts)

    random = RandomizedSearchCV(estimator=clf, param_distributions=clf_params, n_iter=random_search_iter,
                                    cv=k_fold, verbose=2,
                                    random_state=42, n_jobs=-1, scoring=make_scorer(custom_acc))

    if isinstance(clf,EllipticEnvelope):
        random.fit(X_all_tfidf.toarray(), Y)
    else:
        random.fit(X_all_tfidf, Y)

    p = dict(random.best_params_)
    p["score"] = random.best_score_

    return p

def main():

    X = read_data()

    Y = read_labels()

    isf = IsolationForest()
    lof = LocalOutlierFactor(novelty=True)
    svm = OneClassSVM(kernel="rbf")
    cov = EllipticEnvelope()
    kmn = KMeans(n_clusters=1)

    k_fold = StratifiedKFold (n_splits=3, shuffle=True)

    params_isf = []
    params_lof = []
    params_svm = []
    params_cov = []
    params_kmn = []

    for user in range(0,num_of_labeled_users):
        X_all = X[user]
        Y_all = Y[user].astype(int)

        X_genuine = X[user][0:num_of_genuine_segments]
        Y_genuine = Y[user][0:num_of_genuine_segments].astype(int)

        X_unlabeled = X[user][num_of_genuine_segments:]
        Y_unlabeled = Y[user][num_of_genuine_segments:].astype(int)
        '''
        count_vect = CountVectorizer()
        tfidf_transformer = TfidfTransformer(use_idf=False)

        X_all_counts = count_vect.fit_transform(X_all)
        X_all_tfidf = tfidf_transformer.fit_transform(X_all_counts)


        isf_random = RandomizedSearchCV(estimator=isf, param_distributions=ISF_HYPER_PARAMS, n_iter=random_search_iter, cv=k_fold, verbose=2,
                                       random_state=42, n_jobs=-1, scoring=make_scorer(custom_acc))

        svm_random = RandomizedSearchCV(estimator=svm, param_distributions=SVM_HYPER_PARAMS, n_iter=random_search_iter, cv=k_fold, verbose=2,
                                        random_state=42, n_jobs=-1, scoring=make_scorer(custom_acc))

        lof_random = RandomizedSearchCV(estimator=lof, param_distributions=LOF_HYPER_PARAMS, n_iter=random_search_iter, cv=k_fold, verbose=2,
                                        random_state=42, n_jobs=-1, scoring=make_scorer(custom_acc))

        kmn_random = RandomizedSearchCV(estimator=kmn, param_distributions=KMN_HYPER_PARAMS, n_iter=random_search_iter, cv=k_fold, verbose=2,
                                        random_state=42, n_jobs=-1, scoring=make_scorer(custom_acc))

        cov_random = RandomizedSearchCV(estimator=cov, param_distributions=COV_HYPER_PARAMS, n_iter=random_search_iter, cv=k_fold, verbose=2,
                                        random_state=42, n_jobs=-1, scoring=make_scorer(custom_acc))

        isf_random.fit(X_all_tfidf, Y_all)
        svm_random.fit(X_all_tfidf, Y_all)
        lof_random.fit(X_all_tfidf, Y_all)
        kmn_random.fit(X_all_tfidf, Y_all)
        cov_random.fit(X_all_tfidf.toarray(), Y_all)

        p_isf = dict(isf_random.best_params_)
        p_svm = dict(svm_random.best_params_)
        p_lof = dict(lof_random.best_params_)
        p_kmn = dict(kmn_random.best_params_)
        p_cov = dict(cov_random.best_params_)

        p_isf["score"] = isf_random.best_score_
        p_svm["score"] = svm_random.best_score_
        p_lof["score"] = lof_random.best_score_
        p_kmn["score"] = kmn_random.best_score_
        p_cov["score"] = cov_random.best_score_
        
        params_isf.append(p_isf)
        params_svm.append(p_svm)
        params_lof.append(p_lof)
        params_kmn.append(p_kmn)
        params_cov.append(p_cov)
        '''

        params_isf.append(calc_best_detector_for_algoritm(X_unlabeled, Y_unlabeled, isf, ISF_HYPER_PARAMS, k_fold))
        params_svm.append(calc_best_detector_for_algoritm(X_unlabeled, Y_unlabeled, svm, SVM_HYPER_PARAMS, k_fold))
        params_lof.append(calc_best_detector_for_algoritm(X_unlabeled, Y_unlabeled, lof, LOF_HYPER_PARAMS, k_fold))
        params_kmn.append(calc_best_detector_for_algoritm(X_unlabeled, Y_unlabeled, kmn, KMN_HYPER_PARAMS, k_fold))
        params_cov.append(calc_best_detector_for_algoritm(X_unlabeled, Y_unlabeled, cov, COV_HYPER_PARAMS, k_fold))

    write_output(params_isf, 'IsolationForest')
    write_output(params_svm, 'OneClassSVM')
    write_output(params_lof, 'LocalOutlierFactor')
    write_output(params_kmn, 'KMeans')
    write_output(params_cov, 'EllipticEnvelope')


if __name__ == '__main__':
    main()

