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
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn import metrics

output_folder_name = 'Output'
data_folder_name = 'FraudedRawData'
label_file_name = 'challengeToFill.csv'
metrics_file_name = 'metrics.csv'

fieldnames = [
        'rep',
        'learning_rate',
        'steps',
        'classifier',
        'feature_column',
        'train_auc_precision_recall',
        'train_recall',
        'train_average_loss',
        'train_accuracy_baseline',
        'train_accuracy',
        'train_auc',
        'train_loss',
        'train_label/mean',
        'train_precision',
        'train_global_step',
        'train_prediction/mean',
        'test_auc_precision_recall',
        'test_recall',
        'test_average_loss',
        'test_accuracy_baseline',
        'test_accuracy',
        'test_auc',
        'test_loss',
        'test_label/mean',
        'test_precision',
        'test_global_step',
        'test_prediction/mean'
        ]

num_of_users = 40
num_of_labeled_users = 10
num_of_unlabeled_users = 30
unlabeled_users_start = 10

labeled_users = range(0, num_of_labeled_users)
semi_labeled_users = range(0, num_of_users)
unlabeled_users = range(num_of_labeled_users, num_of_users)
all_users = range(0, num_of_users)

num_of_seg = 150
labeled_seg_start = 50
unlabeled_seg_start = 50
num_of_words_in_seg = 100
num_of_labeled_seg = (num_of_seg-labeled_seg_start)*(num_of_users-num_of_labeled_users)


labeled_segments = range(labeled_seg_start, num_of_seg)
semi_labeled_segments = range(0, labeled_seg_start)
unlabeled_segments = range(labeled_seg_start, num_of_seg)
all_segments = range(0, num_of_seg)


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



def read_train_data(labels):

    X = []
    Y = []

    for user in labeled_users:

        user_file = os.path.abspath(os.path.join(data_folder_name, 'User' + str(user)))

        user_df = pd.read_csv(user_file, header=None, names=['Vocalbulary'])

        for seg in all_segments:
            user_seg = user_df['Vocalbulary'][seg * num_of_words_in_seg:(seg + 1) * num_of_words_in_seg].tolist()

            X.append(' '.join(user_seg))

            Y.append((int(labels.values[user, seg])))

    return np.array(X),  np.array(Y)


def main():


    reps = range(4)
    steps_to_try = [500, 1000, 2000]
    embedding_dims_to_try = [2, 4, 10, 100]
    learning_rates_to_try = [0.1, 0.01]
    hidden_units_to_try = [[4, 4], [10, 10], [20, 20]]

    input_file = os.path.abspath(os.path.join(label_file_name))
    metrics_csv_name = os.path.abspath(os.path.join(output_folder_name, metrics_file_name))

    input_file_df = pd.read_csv(input_file, index_col=0)

    X, Y = read_train_data(input_file_df)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


    with  open(metrics_csv_name, 'w', newline='') as metrics_csv:

        csv_writer = csv.DictWriter(metrics_csv, fieldnames=fieldnames)
        csv_writer.writeheader()

    for rep in reps:
        print('\n@@ Rep: ', rep)

        X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, Y, test_size=0.2)

        for learning_rate in learning_rates_to_try:
            print('\n@@ Learning rate: ', learning_rate)

            for steps in steps_to_try:
                print('\n@@ Steps: ', steps)

                for hidden_units in hidden_units_to_try:
                    print('\n@@ Hidden units: ', hidden_units)

                    # create model
                    model = KerasClassifier(build_fn=create_model, verbose=0, epochs=steps,
                                            nn1=hidden_units[0], nn2=hidden_units[1], lr=learning_rate,
                                            input_shape=X_train.shape[1], output_shape=1)

                    model.fit(X_train.toarray(),y_train)

                    y_pred_train = model.predict(X_test.toarray())
                    y_pred_prob_train = model.predict_proba(X_test.toarray())
                    y_pred_prob_train = [p[1] for p in y_pred_prob_train]

                    y_pred_test = model.predict(X_test.toarray())
                    y_pred_prob_test = model.predict_proba(X_test.toarray())
                    y_pred_prob_test = [p[1] for p in y_pred_prob_test]

                    with  open(metrics_csv_name, 'a', newline='') as metrics_csv:
                        csv_writer = csv.DictWriter(metrics_csv, fieldnames)
                        csv_writer.writerow({
                        'rep': rep,
                        'learning_rate': learning_rate,
                        'steps': steps,
                        'classifier': 'dnn classifier with ' + str(hidden_units) + ' hidden_units',
                        'feature_column': 'indicator column',
                        'train_auc_precision_recall': metrics.precision_recall_curve(y_pred_train, y_pred_prob_train),
                        'train_recall': 0,
                        'train_average_loss': 0,
                        'train_accuracy_baseline': 0,
                        'train_accuracy': 0,
                        'train_auc': 0,
                        'train_loss': 0,
                        'train_label/mean': 0,
                        'train_precision': 0,
                        'train_global_step': 0,
                        'train_prediction/mean': 0,
                        'test_auc_precision_recall': 0,
                        'test_recall': 0,
                        'test_average_loss': 0,
                        'test_accuracy_baseline': 0,
                        'test_accuracy': 0,
                        'test_auc': 0,
                        'test_loss': 0,
                        'test_label/mean': 0,
                        'test_precision': 0,
                        'test_global_step': 0,
                        'test_prediction/mean': 0
                        })

                    # DNN classifier with embedding column

                    for embedding_dim in embedding_dims_to_try:
                        print('\n@@ Embedding dim: ', embedding_dim)

    # print('\n@@ Writing metrics to metrics.json')
    # with open(os.path.abspath(os.path.join('MFDCA-DATA','metrics.json')), 'w') as outfile:
    #     json.dump(metrics, outfile)

    # try:
    #     classifier.train(
    #     input_fn=lambda: _input_fn(train_features, train_labels),
    #     steps=steps)

    #     evaluation_metrics = classifier.evaluate(
    #     input_fn=lambda: _input_fn(train_features, train_labels),
    #     steps=steps)
    #     print("Training set metrics:")
    #     for m in evaluation_metrics:
    #         print (m, evaluation_metrics[rep][m])
    #     print ("---")

    #     evaluation_metrics = classifier.evaluate(
    #     input_fn=lambda: _input_fn(test_features, test_labels),
    #     steps=steps)

    #     print ("Test set metrics:")
    #     for m in evaluation_metrics:
    #         print (m, evaluation_metrics[rep][m])
    #     print ("---")

    # except ValueError as err:
    #     print(err)

    # pprint(classifier.get_variable_names())
    # print(classifier.get_variable_value('dnn/input_from_feature_columns/input_layer/commands_embedding/embedding_weights').shape)

    # embedding_matrix = classifier.get_variable_value('dnn/input_from_feature_columns/input_layer/terms_embedding/embedding_weights')

    # for cmd_index in range(len(vocabulary)):
    #     # Create a one-hot encoding for our term. It has 0s everywhere, except for
    #     # a single 1 in the coordinate that corresponds to that term.
    #     cmd_vector = np.zeros(len(vocabulary))
    #     cmd_vector[cmd_index] = 1
    #     # We'll now project that one-hot vector into the embedding space.
    #     embedding_xy = np.matmul(cmd_vector, embedding_matrix)
    #     plt.text(embedding_xy[0],
    #             embedding_xy[1],
    #             vocabulary[cmd_index])

    #     # Do a little setup to make sure the plot displays nicely.
    #     plt.rcParams["figure.figsize"] = (15, 15)
    #     plt.xlim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
    #     plt.ylim(1.2 * embedding_matrix.min(), 1.2 * embedding_matrix.max())
    #     plt.show()


if __name__ == '__main__':
    main()