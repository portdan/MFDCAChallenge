import pandas as pd
import numpy as np
import os
import shutil
import scipy
import csv

import pandas as pd
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from tensorflow.keras.metrics import Accuracy,AUC, Precision, Recall

from sklearn.utils import resample


output_folder_name = 'Output'
data_folder_name = 'FraudedRawData'
label_file_name = 'challengeToFill.csv'
metrics_file_name = 'metrics.csv'

fieldnames = [
        'rep',
        'learning_rate',
        'epochs',
        'classifier',
        'train_average_precision',
        'train_recall',
        'train_log_loss',
        'train_balanced_accuracy',
        'train_accuracy',
        'train_auc',
        'train_roc_auc',
        'train_loss',
        'train_label/mean',
        'train_precision',
        'train_global_step',
        'train_prediction/mean',
        'test_average_precision',
        'test_recall',
        'test_log_loss',
        'test_balanced_accuracy',
        'test_accuracy',
        'test_auc',
        'test_roc_auc',
        'test_loss',
        'test_label/mean',
        'test_precision',
        'test_global_step',
        'test_prediction/mean',
        ]




num_of_labeled_users = 10
num_of_segments = 150
num_of_words_in_seg = 100


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

    return np.array(X),  np.array(Y)

def create_model(nn1=20, nn2=20, input_shape=1, output_shape=2, lr=1e-1):

    # use AdagradOptimizer with gradient clipings
    my_optimizer = keras.optimizers.Adagrad(learning_rate=lr, clipnorm=5)

    # create model
    model = Sequential()
    model.add(Dense(nn1, input_dim=input_shape, activation='relu'))
    model.add(Dense(nn2, activation='relu'))
    model.add(Dense(output_shape, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=my_optimizer)
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


    reps = range(4)
    epochs_to_try = [500, 1000, 2000]
    embedding_dims_to_try = [2, 4, 10, 100]
    learning_rates_to_try = [0.1, 0.01]
    hidden_units_to_try = [[4, 4], [10, 10], [20, 20]]

    input_file = os.path.abspath(os.path.join(label_file_name))
    metrics_csv_name = os.path.abspath(os.path.join(output_folder_name, metrics_file_name))

    input_file_df = pd.read_csv(input_file, index_col=0)

    X, Y = read_train_data(input_file_df)

    # X, Y = upsample(X, Y)


    tokenizer = keras.preprocessing.text.Tokenizer()

    tokenizer.fit_on_texts(X)
    X = tokenizer.texts_to_sequences(X)

    num_classes = max(Y) + 1
    print('# of Classes: {}'.format(num_classes))

    index_to_word = {}
    for key, value in tokenizer.word_index.items():
        index_to_word[value] = key

    print(' '.join([index_to_word[x] for x in X[0]]))
    print(Y[0])

    '''
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    '''
    '''
    X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, Y, test_size=0.2)
    vectorizer = CountVectorizer()
    vectorizer.fit(X_train)
    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)
    '''

    with  open(metrics_csv_name, 'w', newline='') as metrics_csv:
        csv_writer = csv.DictWriter(metrics_csv, fieldnames=fieldnames)
        csv_writer.writeheader()

    for rep in reps:
        print('\n@@ Rep: ', rep)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        max_words = 10000

        tokenizer = keras.preprocessing.text.Tokenizer().fi
        X_train = tokenizer.sequences_to_matrix(X_train, mode='tfidf')
        X_test = tokenizer.sequences_to_matrix(X_test, mode='tfidf')

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        for learning_rate in learning_rates_to_try:
            print('\n@@ Learning rate: ', learning_rate)

            for epochs in epochs_to_try:
                print('\n@@ Epochs: ', epochs)

                for hidden_units in hidden_units_to_try:
                    print('\n@@ Hidden units: ', hidden_units)

                    model = Sequential()
                    model.add(Dense(512, input_shape=(max_words,)))
                    model.add(Activation('relu'))
                    model.add(Dropout(0.5))
                    model.add(Dense(num_classes))
                    model.add(Activation('softmax'))

                    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[Accuracy(), AUC(), Precision(), Recall()])
                    print(model.metrics_names)

                    batch_size = 32
                    epochs = 3

                    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                                        validation_split=0.1, class_weight = {0:10, 1:1})
                    score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)

                    print('Test loss:', score[0])
                    print('Test accuracy:', score[1])
                    print('Test auc:', score[2])
                    print('Test precision:', score[3])
                    print('Test recall:', score[4])

                    y_pred_train = model.predict(X_train.toarray())
                    y_pred_prob_train = model.predict_proba(X_train.toarray())
                    y_pred_prob_train = [p[1] for p in y_pred_prob_train]

                    y_pred_test = model.predict(X_test.toarray())
                    y_pred_prob_test = model.predict_proba(X_test.toarray())
                    y_pred_prob_test = [p[1] for p in y_pred_prob_test]

                    with open(metrics_csv_name, 'a', newline='') as metrics_csv:
                        csv_writer = csv.DictWriter(metrics_csv, fieldnames)
                        csv_writer.writerow({
                        'rep': rep,
                        'learning_rate': learning_rate,
                        'epochs': epochs,
                        'classifier': 'dnn classifier with ' + str(hidden_units) + ' hidden_units',
                        'train_average_precision': metrics.average_precision_score(y_pred_train, y_pred_prob_train),
                        'train_recall': metrics.recall_score(y_pred_train, y_train),
                        'train_log_loss': metrics.log_loss(y_pred_train, y_pred_prob_train),
                        'train_balanced_accuracy': metrics.balanced_accuracy_score(y_pred_train, y_train),
                        'train_accuracy': metrics.accuracy_score(y_pred_train, y_train),
                        'train_auc': 0,
                        'train_roc_auc': metrics.roc_auc_score(y_pred_train, y_pred_prob_train),
                        'train_loss': 0,
                        'train_label/mean': 0,
                        'train_precision': metrics.precision_score(y_pred_train, y_train),
                        'train_global_step': 0,
                        'train_prediction/mean': 0,
                        'test_average_precision': metrics.average_precision_score(y_pred_test, y_pred_prob_test),
                        'test_recall': metrics.recall_score(y_pred_test, y_test),
                        'test_log_loss': metrics.log_loss(y_pred_test, y_pred_prob_test),
                        'test_balanced_accuracy': metrics.balanced_accuracy_score(y_pred_test, y_test),
                        'test_accuracy': metrics.accuracy_score(y_pred_test, y_test),
                        'test_auc': 0,
                        'test_roc_auc': metrics.roc_auc_score(y_pred_test, y_pred_prob_test),
                        'test_loss': 0,
                        'test_label/mean': 0,
                        'test_precision': metrics.precision_score(y_pred_test, y_test),
                        'test_global_step': 0,
                        'test_prediction/mean': 0,
                        })

                    # DNN classifier with embedding column

                    for embedding_dim in embedding_dims_to_try:
                        print('\n@@ Embedding dim: ', embedding_dim)

if __name__ == '__main__':
    main()