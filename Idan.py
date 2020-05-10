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


output_folder_name = 'Output'
data_folder_name = 'FraudedRawData'
label_file_name = 'challengeToFill.csv'
metrics_file_name = 'metrics.csv'

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

def train_and_test_classifier(classifier, steps, train_features, train_labels, test_features, test_labels):
    metrics = {}
    try:
        classifier.train(
            input_fn=lambda: _input_fn(train_features, train_labels),
            steps=steps)

        metrics['train_metrics'] = classifier.evaluate(
            input_fn=lambda: _input_fn(train_features, train_labels),
            steps=steps)
        print("Training set metrics:")
        for key, val in metrics['train_metrics'].items():
            print(key, val)
        print("---")

        metrics['test_metrics'] = classifier.evaluate(
            input_fn=lambda: _input_fn(test_features, test_labels),
            steps=steps)

        print("Test set metrics:")
        for key, val in metrics['test_metrics'].items():
            print(key, val)
        print("---")

    except ValueError as err:
        print(err)

    return metrics


def main():


    reps = range(4)
    steps_to_try = [500, 1000, 2000]
    embedding_dims_to_try = [2, 4, 10, 100]
    learning_rates_to_try = [0.1, 0.01]
    hidden_units_to_try = [[4, 4], [10, 10], [20, 20]]
    metrics = {}

    input_file = os.path.abspath(os.path.join(label_file_name))
    metrics_csv_name = os.path.abspath(os.path.join(output_folder_name, metrics_file_name))

    input_file_df = pd.read_csv(input_file, index_col=0)

    X, Y = read_train_data(input_file_df)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


    metrics_csv = open(metrics_csv_name, 'w', newline='')
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
    csv_writer = csv.DictWriter(metrics_csv, fieldnames=fieldnames)
    csv_writer.writeheader()

    for rep in reps:
        print('\n@@ Rep: ', rep)
        metrics[rep] = {}

        X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, Y, test_size=0.2)

        for learning_rate in learning_rates_to_try:
            print('\n@@ Learning rate: ', learning_rate)
            metrics[rep][learning_rate] = {}
            for steps in steps_to_try:
                print('\n@@ Steps: ', steps)
                metrics[rep][learning_rate][steps] = {}

                # use AdagradOptimizer with gradient clipings
                my_optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate, clipnorm=5)

                metrics[rep][learning_rate][steps]['dnn_classifier'] = {}
                for hidden_units in hidden_units_to_try:
                    print('\n@@ Hidden units: ', hidden_units)
                    metrics[rep][learning_rate][steps]['dnn_classifier'][str(hidden_units)] = {}

                    # create model
                    model = Sequential()
                    model.add(Dense(4, input_dim=X_train.shape[1], activation='relu'))
                    model.add(Dense(1, activation='sigmoid'))
                    # Compile model
                    model.compile(loss='binary_crossentropy', optimizer=my_optimizer)

                    model.fit(X_train,y_train)

                    # DNN classifier with indicator column
                    commands_indicator_column = tf.feature_column.indicator_column(commands_feature_column)
                    feature_columns = [commands_indicator_column]

                    classifier = tf.estimator.DNNClassifier(
                        feature_columns=feature_columns,
                        hidden_units=hidden_units,
                        optimizer=my_optimizer
                    )

                    print('\n@@ Running dnn classifier with indicator column: ')
                    m = train_and_test_classifier(classifier, steps, train_features, train_labels, test_features,
                                                  test_labels)

                    metrics[rep][learning_rate][steps]['dnn_classifier'][str(hidden_units)]['indicator_column'] = m
                    csv_writer.writerow({
                        'rep': rep,
                        'learning_rate': learning_rate,
                        'steps': steps,
                        'classifier': 'dnn classifier with ' + str(hidden_units) + ' hidden_units',
                        'feature_column': 'indicator column',
                        'train_auc_precision_recall': m['train_metrics']['auc_precision_recall'],
                        'train_recall': m['train_metrics']['recall'],
                        'train_average_loss': m['train_metrics']['average_loss'],
                        'train_accuracy_baseline': m['train_metrics']['accuracy_baseline'],
                        'train_accuracy': m['train_metrics']['accuracy'],
                        'train_auc': m['train_metrics']['auc'],
                        'train_loss': m['train_metrics']['loss'],
                        'train_label/mean': m['train_metrics']['label/mean'],
                        'train_precision': m['train_metrics']['precision'],
                        'train_global_step': m['train_metrics']['global_step'],
                        'train_prediction/mean': m['train_metrics']['prediction/mean'],
                        'test_auc_precision_recall': m['test_metrics']['auc_precision_recall'],
                        'test_recall': m['test_metrics']['recall'],
                        'test_average_loss': m['test_metrics']['average_loss'],
                        'test_accuracy_baseline': m['test_metrics']['accuracy_baseline'],
                        'test_accuracy': m['test_metrics']['accuracy'],
                        'test_auc': m['test_metrics']['auc'],
                        'test_loss': m['test_metrics']['loss'],
                        'test_label/mean': m['test_metrics']['label/mean'],
                        'test_precision': m['test_metrics']['precision'],
                        'test_global_step': m['test_metrics']['global_step'],
                        'test_prediction/mean': m['test_metrics']['prediction/mean']
                    })

                    # DNN classifier with embedding column
                    metrics[rep][learning_rate][steps]['dnn_classifier'][str(hidden_units)]['embedding_column'] = {}
                    for embedding_dim in embedding_dims_to_try:
                        print('\n@@ Embedding dim: ', embedding_dim)
                        commands_embedding_column = tf.feature_column.embedding_column(commands_feature_column,
                                                                                       dimension=embedding_dim)
                        feature_columns = [commands_embedding_column]

                        classifier = tf.estimator.DNNClassifier(
                            feature_columns=feature_columns,
                            hidden_units=hidden_units,
                            optimizer=my_optimizer
                        )

                        print('\n@@ Running dnn classifier with embedding column: ')
                        m = train_and_test_classifier(
                            classifier,
                            steps,
                            train_features,
                            train_labels,
                            test_features,
                            test_labels)

                        metrics[rep][learning_rate][steps]['dnn_classifier'][str(hidden_units)]['embedding_column'][
                            str(embedding_dim)] = m
                        csv_writer.writerow({
                            'rep': rep,
                            'learning_rate': learning_rate,
                            'steps': steps,
                            'classifier': 'dnn classifier with ' + str(hidden_units) + ' hidden_units',
                            'feature_column': str(embedding_dim) + ' dim embedding column',
                            'train_auc_precision_recall': m['train_metrics']['auc_precision_recall'],
                            'train_recall': m['train_metrics']['recall'],
                            'train_average_loss': m['train_metrics']['average_loss'],
                            'train_accuracy_baseline': m['train_metrics']['accuracy_baseline'],
                            'train_accuracy': m['train_metrics']['accuracy'],
                            'train_auc': m['train_metrics']['auc'],
                            'train_loss': m['train_metrics']['loss'],
                            'train_label/mean': m['train_metrics']['label/mean'],
                            'train_precision': m['train_metrics']['precision'],
                            'train_global_step': m['train_metrics']['global_step'],
                            'train_prediction/mean': m['train_metrics']['prediction/mean'],
                            'test_auc_precision_recall': m['test_metrics']['auc_precision_recall'],
                            'test_recall': m['test_metrics']['recall'],
                            'test_average_loss': m['test_metrics']['average_loss'],
                            'test_accuracy_baseline': m['test_metrics']['accuracy_baseline'],
                            'test_accuracy': m['test_metrics']['accuracy'],
                            'test_auc': m['test_metrics']['auc'],
                            'test_loss': m['test_metrics']['loss'],
                            'test_label/mean': m['test_metrics']['label/mean'],
                            'test_precision': m['test_metrics']['precision'],
                            'test_global_step': m['test_metrics']['global_step'],
                            'test_prediction/mean': m['test_metrics']['prediction/mean']
                        })

    metrics_csv.close()

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