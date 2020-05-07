import pandas as pd
import numpy as np
import os
import shutil
import scipy

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics


data_folder_name = 'FraudedRawData'
label_file_name = 'challengeToFill.csv'


labeled_users = range(0,10)
semi_labeled_users = range(0,40)
unlabeled_users = range(10,40)

num_of_seg = 150
num_of_words_in_seg = 100
num_of_freq_words = 150

labeled_segments = range(50,150)
semi_labeled_segments = range(0,50)
unlabeled_segments = range(50,150)

def main():

    label_file = os.path.abspath(os.path.join(label_file_name))
    labels = pd.read_csv(label_file,index_col=0)

    X_labeled = []
    X_semi_labeled = []
    X_unlabeled = []

    Y_labeled = []
    Y_semi_labeled = []

    read_labeled_data(X_labeled, Y_labeled, labels)

    read_semi_labeled_data(X_semi_labeled, Y_semi_labeled, labels)

    read_unlabeled_data(X_unlabeled)

    X = X_labeled
    Y = Y_labeled

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, Y, test_size=0.2)

    clf = MultinomialNB().fit(X_train, y_train)

    clf2 = SGDClassifier(loss='hinge', penalty='l2',
                              alpha=1e-3, random_state=42,
                              max_iter=5, tol=None).fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d"
          % (X_test.shape[0], (y_test != y_pred).sum()))
    Accuracy = np.mean(y_pred == y_test) * 100
    print("Accuracy : %f" % Accuracy)

    print(metrics.classification_report(y_test, y_pred))

    y_pred = clf2.predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d"
          % (X_test.shape[0], (y_test != y_pred).sum()))
    Accuracy = np.mean(y_pred == y_test) * 100
    print("Accuracy : %f" % Accuracy)

    print(metrics.classification_report(y_test, y_pred))


    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                              alpha=1e-3, random_state=42,
                              max_iter=5, tol=None)),
    ])

    X_new_counts = count_vect.transform(X_unlabeled)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    y_pred = clf.predict(X_new_tfidf)

    y_pred = np.reshape(y_pred,(30,100))

    new_labels = labels.values

    new_labels[10:40, 50:150] = y_pred

    new_labels_df =  pd.DataFrame(data=new_labels, index =labels.index, columns = labels.columns, dtype=int)

    new_labels_df.to_csv('test' + label_file_name, index=True)


def read_unlabeled_data(X):

    for user in unlabeled_users:

        user_file = os.path.abspath(os.path.join(data_folder_name, 'User' + str(user)))

        user_df = pd.read_csv(user_file, header=None, names=['Vocalbulary'])

        for seg in unlabeled_segments:
            user_seg = user_df['Vocalbulary'][seg * num_of_words_in_seg:(seg + 1) * num_of_words_in_seg].tolist()

            X.append(' '.join(user_seg))



def read_semi_labeled_data(X, Y, labels):

    for user in semi_labeled_users:

        user_file = os.path.abspath(os.path.join(data_folder_name, 'User' + str(user)))

        user_df = pd.read_csv(user_file, header=None, names=['Vocalbulary'])

        for seg in semi_labeled_segments:
            user_seg = user_df['Vocalbulary'][seg * num_of_words_in_seg:(seg + 1) * num_of_words_in_seg].tolist()

            X.append(' '.join(user_seg))

            Y.append(int(labels.values[user, seg]))


def read_labeled_data(X, Y, labels):

    for user in labeled_users:

        user_file = os.path.abspath(os.path.join(data_folder_name, 'User' + str(user)))

        user_df = pd.read_csv(user_file, header=None, names=['Vocalbulary'])

        for seg in labeled_segments:
            user_seg = user_df['Vocalbulary'][seg * num_of_words_in_seg:(seg + 1) * num_of_words_in_seg].tolist()

            X.append(' '.join(user_seg))

            Y.append(int(labels.values[user, seg]))


if __name__ == '__main__':
    main()