import numpy as np
import os
import csv

import pandas as pd
from tensorflow import keras

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix, roc_auc_score, auc

from sklearn.utils import resample
from sklearn.utils import class_weight

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import AllKNN

output_folder_name = 'Output'
data_folder_name = 'FraudedRawData'
label_file_name = 'challengeToFill.csv'
metrics_file_name = 'metrics.csv'

fieldnames = [
    'rep',
    'learning_rate',
    'epochs',
    'hidden_units',
    'optimizer',
    'loss_function',
    'train_accuracy',
    'train_precision',
    'train_recall',
    'train_f1_score',
    'train_roc_auc_score',
    'train_auc',
    'train_confusion_matrix',
    'test_accuracy',
    'test_precision',
    'test_recall',
    'test_f1_score',
    'test_roc_auc_score',
    'test_auc',
    'test_confusion_matrix',
]

num_of_labeled_users = 10
num_of_segments = 150
num_of_words_in_seg = 100

Reps = range(1)

# Epochs = [500, 1000, 2000]
# Learning_rates = [1e-1, 1e-2, 1e-3]
# Hidden_units = [[4, 4, 0], [10, 10, 0], [50, 50, 0], [100, 100, 0], [10, 10, 10], [50, 50, 50], [100, 100, 100]]
# Optimizers = ['Adam', 'Adagrad', 'SGD']
# Loss_functions = ['binary_crossentropy', 'hinge', 'squared_hinge']

Epochs = [5000, 10000, 50000, 100000]
Learning_rates = [1e-1, 1e-2]
Hidden_units = [[200, 200, 200], [500, 500, 500] ,[1000, 1000, 1000]]
Optimizers = ['Adam', 'Adagrad', 'SGD']
Loss_functions = ['binary_crossentropy']

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


def upsample(X, Y):
    X_vec = np.array(X).reshape(-1, 1)
    Y_vec = np.array(Y).reshape(-1, 1)

    res = np.column_stack((X_vec, Y_vec))
    res_df = pd.DataFrame(data=res, columns=["Segment", "Label"])

    # Separate majority and minority classes
    res_majority = res_df[res_df['Label'].astype(int) == 0]
    res_minority = res_df[res_df['Label'].astype(int) == 1]

    # Upsample minority class
    res_minority_upsampled = resample(res_minority,
                                      replace=True,  # sample with replacement
                                      n_samples=res_majority.shape[0])  # to match majority class

    # Combine majority class with upsampled minority class
    res_upsampled = pd.concat([res_majority, res_minority_upsampled])

    return np.array(res_upsampled['Segment']), np.array(res_upsampled['Label']).astype(int)


def main():
    input_file = os.path.abspath(os.path.join(label_file_name))
    metrics_csv_name = os.path.abspath(os.path.join(output_folder_name, metrics_file_name))

    input_file_df = pd.read_csv(input_file, index_col=0)

    X, Y = read_train_data(input_file_df)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train.reshape(-1))
    X_test_counts = count_vect.transform(X_test)

    tfidf_transformer = TfidfTransformer()
    X_train = tfidf_transformer.fit_transform(X_train_counts)
    X_test = tfidf_transformer.transform(X_test_counts)

    # X_train, y_train = upsample(X_train, y_train)

    # ros = RandomOverSampler(random_state=777)
    # X_train, y_train = ros.fit_sample(X_train, y_train)

    # sm = SMOTE(random_state=777, sampling_strategy=1.0)
    # X_train, y_train = sm.fit_sample(X_train, y_train)

    # sampler = AllKNN(allow_minority=True, n_neighbors=20)
    # X_train, y_train = sampler.fit_sample(X_train, y_train)
    #
    # sm = SMOTE(random_state=777, sampling_strategy=1.0)
    # X_train, y_train = sm.fit_sample(X_train, y_train)

    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

    with open(metrics_csv_name, 'w', newline='') as metrics_csv:
        csv_writer = csv.DictWriter(metrics_csv, fieldnames=fieldnames)
        csv_writer.writeheader()

    for rep in Reps:
        print('\n@@ Rep: ', rep)

        for learning_rate in Learning_rates:
            print('\n@@ Learning rate: ', learning_rate)

            for epochs in Epochs:
                print('\n@@ Epochs: ', epochs)

                for hidden_units in Hidden_units:
                    print('\n@@ hidden units: ', hidden_units)

                    for optimizer in Optimizers:
                        print('\n@@ optimizer: ', optimizer)

                        for loss_function in Loss_functions:
                            print('\n@@ loss_function: ', loss_function)

                            # create model
                            model = KerasClassifier(build_fn=create_model, verbose=0,
                                                    nn1=hidden_units[0], nn2=hidden_units[1], nn3=hidden_units[2],
                                                    optimizer=optimizer, loss=loss_function,
                                                    activation='relu', input_shape=X_train.shape[1], output_shape=1,
                                                    lr=learning_rate)

                            # model.fit(X_train.toarray(), y_train, class_weight=class_weights)
                            model.fit(X_train.toarray(), y_train)

                            y_pred_train = model.predict(X_train.toarray())
                            y_pred_prob_train = model.predict_proba(X_train.toarray())
                            y_pred_prob_train = [p[1] for p in y_pred_prob_train]

                            y_pred_test = model.predict(X_test.toarray())
                            y_pred_prob_test = model.predict_proba(X_test.toarray())
                            y_pred_prob_test = [p[1] for p in y_pred_prob_test]

                            train_accuracy = accuracy_score(y_pred_train, y_train)
                            train_precision = precision_score(y_pred_train, y_pred_train, zero_division=1)
                            train_recall = recall_score(y_pred_train, y_train, zero_division=1)
                            train_f1_score = f1_score(y_pred_train, y_train, zero_division=1)

                            try:
                                train_roc_auc_score = roc_auc_score(y_pred_train, y_pred_prob_train)
                            except:
                                train_roc_auc_score = -1
                            try:
                                train_auc = auc(y_pred_train, y_pred_prob_train)
                            except:
                                train_auc = -1

                            train_confusion_matrix = confusion_matrix(y_pred_train, y_train)

                            test_accuracy = accuracy_score(y_pred_test, y_test)
                            test_precision = precision_score(y_pred_test, y_pred_test, zero_division=1)
                            test_recall = recall_score(y_pred_test, y_test, zero_division=1)
                            test_f1_score = f1_score(y_pred_test, y_test, zero_division=1)

                            try:
                                test_roc_auc_score = roc_auc_score(y_pred_test, y_pred_prob_test)
                            except:
                                test_roc_auc_score = -1
                            try:
                                test_auc = auc(y_pred_test, y_pred_prob_test)
                            except:
                                test_auc = -1

                            test_confusion_matrix = confusion_matrix(y_pred_test, y_test)

                            with open(metrics_csv_name, 'a', newline='') as metrics_csv:
                                csv_writer = csv.DictWriter(metrics_csv, fieldnames)
                                csv_writer.writerow({
                                    'rep': rep,
                                    'learning_rate': learning_rate,
                                    'epochs': epochs,
                                    'hidden_units': hidden_units,
                                    'optimizer': optimizer,
                                    'loss_function': loss_function,
                                    'train_accuracy': train_accuracy,
                                    'train_precision': train_precision,
                                    'train_recall': train_recall,
                                    'train_f1_score': train_f1_score,
                                    'train_roc_auc_score': train_roc_auc_score,
                                    'train_auc': train_auc,
                                    'train_confusion_matrix': train_confusion_matrix,
                                    'test_accuracy': test_accuracy,
                                    'test_precision': test_precision,
                                    'test_recall': test_recall,
                                    'test_f1_score': test_f1_score,
                                    'test_roc_auc_score': test_roc_auc_score,
                                    'test_auc': test_auc,
                                    'test_confusion_matrix': test_confusion_matrix
                                })

if __name__ == '__main__':
    main()
