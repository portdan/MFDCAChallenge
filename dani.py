import numpy as np
import os
import csv
import shutil

import pandas as pd
from tensorflow import keras

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix, roc_auc_score, auc

from sklearn.utils import resample
from sklearn.utils import class_weight

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import AllKNN

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

def read_train_data(labels):
    X = []
    Y = []

    for user in range(0, num_of_labeled_users):

        user_file = os.path.abspath(os.path.join(data_folder_name, 'User' + str(user)))

        user_df = pd.read_csv(user_file, header=None, names=['Vocalbulary'])

        for seg in range(0, num_of_segments):
            user_seg = user_df['Vocalbulary'][seg * num_of_words_in_seg:(seg + 1) * num_of_words_in_seg].tolist()

            X.append(' '.join(user_seg))

            Y.append((int(labels.values[user, seg])))

    return np.array(X), np.array(Y)

def read_unlabeled_data():

    X_unlabeled = np.empty((num_of_users - num_of_labeled_users, num_of_segments-num_of_genuine_segments),dtype=object)

    for user in range(num_of_labeled_users, num_of_users):

        user_file = os.path.abspath(os.path.join(data_folder_name, 'User' + str(user)))

        user_df = pd.read_csv(user_file, header=None, names=['Vocalbulary'])

        user_segments = np.empty((1,num_of_segments-num_of_genuine_segments),dtype=object)

        for seg in range(num_of_genuine_segments, num_of_segments):
            user_seg_str = user_df['Vocalbulary'][seg * num_of_words_in_seg:(seg + 1) * num_of_words_in_seg].tolist()
            user_segments[:,seg-num_of_genuine_segments] = ' '.join(user_seg_str)

        X_unlabeled[user - num_of_labeled_users] = user_segments

    return X_unlabeled


def create_model(nn1=20, nn2=20, nn3=20, optimizer='Adam', loss='binary_crossentropy',
                 activation='relu', input_shape=1, output_shape=2, lr=1e-2):
    if optimizer is 'Adam':
        optimizer = keras.optimizers.Adam(learning_rate=lr)
    if optimizer is 'Adagrad':
        optimizer = keras.optimizers.Adagrad(learning_rate=lr)
    if optimizer is 'SGD':
        optimizer = keras.optimizers.SGD(learning_rate=lr)

    # create model
    model = Sequential()

    model.add(Dense(nn1, input_dim=input_shape, activation=activation))

    if nn2 > 0:
        model.add(Dense(nn2, activation=activation))
    if nn3 > 0:
        model.add(Dense(nn3, activation=activation))

    model.add(Dense(output_shape, activation='sigmoid'))

    # Compile model
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

def WriteOutput(input_file_df, y_pred):
    output_df = input_file_df.values

    output_df[num_of_labeled_users:num_of_users, num_of_genuine_segments:num_of_segments] = y_pred

    output_df = pd.DataFrame(data=output_df, index=input_file_df.index, columns=input_file_df.columns, dtype=int)

    output_file = os.path.abspath(os.path.join(output_folder_name, label_file_name))

    if not os.path.exists(os.path.abspath(output_folder_name)):
        shutil.rmtree(os.path.abspath(output_folder_name))
    os.makedirs(os.path.abspath(output_folder_name))

    output_df.to_csv(output_file, index=True)

def ClassifyPerUser(X_unlabeled, count_vect, tfidf_transformer, clf):

    y_pred = np.zeros_like(X_unlabeled)

    for user in range(0, X_unlabeled.shape[0]):
        user_seg = X_unlabeled[user]

        X_new_counts = count_vect.transform(user_seg)
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)

        pred_prob = clf.predict_proba(X_new_tfidf.toarray())
        top_ten_pred = np.argsort(pred_prob[:, 0])[0:10]

        malicious = np.zeros_like(user_seg, dtype=int)
        malicious[top_ten_pred] = 1

        y_pred[user] = malicious

    return y_pred


def main():
    input_file = os.path.abspath(os.path.join(label_file_name))

    input_file_df = pd.read_csv(input_file, index_col=0)

    X, Y = read_train_data(input_file_df)
    X_unlabeled = read_unlabeled_data()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train.reshape(-1))
    X_test_counts = count_vect.transform(X_test)

    tfidf_transformer = TfidfTransformer()
    X_train = tfidf_transformer.fit_transform(X_train_counts)
    X_test = tfidf_transformer.transform(X_test_counts)

    # create model
    model = KerasClassifier(build_fn=create_model, verbose=0,
                            nn1=Hidden_units[0], nn2=Hidden_units[1], nn3=Hidden_units[2],
                            optimizer=Optimizer, loss=Loss_function,
                            activation='relu', input_shape=X_train.shape[1], output_shape=1,
                            lr=Learning_rate)

    model.fit(X_train.toarray(), y_train)

    y_pred = ClassifyPerUser(X_unlabeled, count_vect, tfidf_transformer, model)

    WriteOutput(input_file_df, y_pred)

if __name__ == '__main__':
    main()

