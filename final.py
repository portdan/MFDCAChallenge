import numpy as np
import os
import csv
import shutil

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import regularizers

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope






output_folder_name = 'Output2'
data_folder_name = 'FraudedRawData'
label_file_name = 'challengeToFill.csv'

num_of_users = 40
num_of_labeled_users = 10
num_of_segments = 150
num_of_genuine_segments = 50
num_of_words_in_seg = 100

Epochs = 5000
Learning_rate = 1e-2
Hidden_units = [1000, 1000, 1000]
Optimizer = 'Adam'
Loss_function = 'binary_crossentropy'

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

def create_autoencoder(nn1=20, nn2=20, nn3=20, optimizer='Adam', loss='binary_crossentropy',
                 activation='relu', input_shape=1, output_shape=2, lr=1e-2, encoding_dim=150):
    # Create model
    input_layer = Input(shape=(input_shape,))
    encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoder = Dense(int(encoding_dim / 2), activation="tanh")(encoder)
    encoder = Dense(int(2), activation="tanh")(encoder)
    decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
    decoder = Dense(int(encoding_dim), activation='tanh')(decoder)
    decoder = Dense(input_shape, activation='tanh')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    # Compile model
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

def main():

    input_file = os.path.abspath(os.path.join(label_file_name))

    input_file_df = pd.read_csv(input_file, index_col=0)

    X = read_data()

    Y = read_labels()

    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    isf = IsolationForest()
    lof = LocalOutlierFactor(novelty=True)
    svm = OneClassSVM()
    cov = EllipticEnvelope()

    preds_isf = []
    preds_lof = []
    preds_svm = []
    preds_cov =[]
    preds_ae =[]

    preds= []

    for user in range(0,num_of_users):
        X_all = X[user]
        X_labeled = X[user][0:num_of_genuine_segments]
        X_unlabeled = X[user][num_of_genuine_segments:]

        count_vect = CountVectorizer()
        tfidf_transformer = TfidfTransformer()

        X_all_counts = count_vect.fit_transform(X_all)
        X_labeled_counts = count_vect.transform(X_labeled)
        X_unlabeled_counts = count_vect.transform(X_unlabeled)

        X_all_tfidf = tfidf_transformer.fit_transform(X_all_counts)
        X_labeled_tfidf = tfidf_transformer.transform(X_labeled_counts)
        X_unlabeled_tfidf = tfidf_transformer.transform(X_unlabeled_counts)

        ae = create_autoencoder(input_shape=X_all_tfidf.shape[1])

        isf.fit(X_all_tfidf)
        lof.fit(X_all_tfidf)
        svm.fit(X_all_tfidf)
        cov.fit(X_all_tfidf.toarray())
        ae.fit(X_all_tfidf.toarray(), X_all_tfidf.toarray(),
                        epochs=100,
                        batch_size=50,
                        verbose=0
                        )

        pred_isf = isf.predict(X_unlabeled_tfidf)
        pred_lof = lof.predict(X_unlabeled_tfidf)
        pred_svm = svm.predict(X_unlabeled_tfidf)
        pred_cov = cov.predict(X_unlabeled_tfidf.toarray())
        pred_ae = ae.predict(X_unlabeled_tfidf.toarray())

        pred_isf = [1 if p == -1 else 0 for p in pred_isf]
        pred_lof = [1 if p == -1 else 0 for p in pred_lof]
        pred_svm = [1 if p == -1 else 0 for p in pred_svm]
        pred_cov = [1 if p == -1 else 0 for p in pred_cov]

        mse = np.mean(np.power(X_unlabeled_tfidf - pred_ae, 2), axis=1)
        mae = np.mean(np.abs(X_unlabeled_tfidf - pred_ae), axis=1)
        pred_ae = [1 if p == -1 else 0 for p in mae > 0.035]

        preds_lof.append(pred_lof)
        preds_isf.append(pred_isf)
        preds_svm.append(pred_svm)
        preds_cov.append(pred_cov)
        preds_ae.append(pred_ae)

        pred_sum = np.array(pred_lof) + np.array(pred_isf) + np.array(pred_svm) + np.array(pred_cov) + np.array(pred_ae)
        majority = [1 if i > 2 else 0 for i in pred_sum]
        preds.append(majority)


    # create model
    model = KerasClassifier(build_fn=create_model, verbose=0,
                            nn1=Hidden_units[0], nn2=Hidden_units[1], nn3=Hidden_units[2],
                            optimizer=Optimizer, loss=Loss_function,
                            activation='relu', input_shape=X_train.shape[1], output_shape=1,
                            lr=Learning_rate)

    model.fit(X_train.toarray(), y_train)

    y_pred = model.predict(X_unlabeld.toarray())
    y_pred = y_pred.reshape(num_of_users - num_of_labeled_users, num_of_segments-num_of_genuine_segments)

    WriteOutput(input_file_df, y_pred)

if __name__ == '__main__':
    main()

