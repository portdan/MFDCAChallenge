import pandas as pd
import numpy as np
import os
import shutil

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

preprocessed_folder = 'PreProcessedData'
#https://scikit-learn.org/stable/tutorial/machine_learning_map

num_of_users = 10
num_of_seg = 150
num_of_words_in_seg = 100
num_of_freq_words = 150

def main():

    users = ['User' + str(i) for i in range(num_of_users)]

    input_file = os.path.abspath(os.path.join(preprocessed_folder,"total.csv"))

    df = pd.read_csv(input_file)

    data = df.values

    data = data[0:num_of_freq_words,0]

    X = np.zeros((num_of_users*num_of_seg,num_of_freq_words))

    for user in range(num_of_users):
        for seg in range(0, num_of_seg):
            input_file = os.path.abspath(os.path.join(preprocessed_folder, "User" + str(user), "seg-" + str(seg) + ".csv"))
            df_seg = pd.read_csv(input_file)
            data_seg = df_seg.values[:,0]
            data_seg_counts = df_seg.values[:,1]

            X_seg = np.zeros(num_of_freq_words)
            for i in range(len(data)):
                for j in range(len(data_seg)):
                    if data[i]==data_seg[j]:
                        X[user*num_of_seg + seg,i] = data_seg_counts[j]

    input_file = os.path.abspath(os.path.join("challengeToFill.csv"))
    df = pd.read_csv(input_file)

    Y = df.values[0:10,1:151]
    Y = Y.astype('int32')
    Y = np.reshape(Y,(1500,))

    #X, Y = make_classification(n_features=4, random_state=0)
    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(X, Y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d"
           % (X_test.shape[0], (y_test != y_pred).sum()))
    
    zzz = 1
'''
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    test = np.sum(kmeans.labels_ == np.reshape(Y,(1500,)))
    test2 = np.sum(np.zeros(1500)== np.reshape(Y,(1500,)))

    neigh = NearestNeighbors(n_neighbors=2)
    neigh.fit(X, Y)

    reg = linear_model.LinearRegression()
    reg.fit(X,np.reshape(Y,(1500,)))
'''
    

if __name__ == '__main__':
    main()