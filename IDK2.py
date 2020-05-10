import pandas as pd
import numpy as np
import os
import shutil
import scipy

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, make_scorer
from sklearn.metrics import classification_report

from sklearn.utils import resample

from tensorflow import keras
from tensorflow.keras.metrics import AUC, MeanSquaredError, Accuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from kerastuner.tuners import RandomSearch

output_folder_name = 'Output'
data_folder_name = 'FraudedRawData'
label_file_name = 'challengeToFill.csv'

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

Input_dim=1
Output_dim=1

def main():

    input_file = os.path.abspath(os.path.join(label_file_name))
    input_file_df = pd.read_csv(input_file, index_col=0)

    #X_all, Y_all = read_data(input_file_df)
    X_labeled, Y_labeled = read_train_data(input_file_df)
    #X_semi_labeled, Y_semi_labeled = read_semi_labeled_data(input_file_df)

    # X = np.append(X_labeled, X_semi_labeled)
    # Y = np.append(Y_labeled, Y_semi_labeled)

    X = X_labeled
    Y = Y_labeled

    # create the tokenizer
    tokenizer = keras.preprocessing.text.Tokenizer()
    # fit the tokenizer on the documents
    tokenizer.fit_on_texts(X)
    # integer encode documents
    #encoded_docs = tokenizer.texts_to_matrix(X, mode='tfidf')

    # summarize what was learned
    # print(tokenizer.word_counts)
    # print(tokenizer.document_count)
    # print(tokenizer.word_index)
    # print(tokenizer.word_docs)

    X = tokenizer.texts_to_sequences(X)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_labeled)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


    X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, Y, test_size=0.2)

    global Input_dim
    Input_dim = X_train.shape[1]
    global Output_dim 
    Output_dim = 1

    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=1,
        executions_per_trial=1,
        directory='my_dir',
        project_name='helloworld')

    tuner.search(X_train.toarray(), y_train, epochs=100, validation_data=(X_test.toarray(),y_test))

    tuner.results_summary()


def build_model(hp):

    model = Sequential()

    model.add(Dense(input_dim=Input_dim,
                           units=hp.Int('units',
                                        min_value=20,
                                        max_value=100,
                                        step=20),
                           activation='relu'))

    model.add(Dense(Output_dim, activation='sigmoid'))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-1, 1e-2, 1e-3])),
        loss='binary_crossentropy',
        metrics=['accuracy', 'categorical_accuracy', 'binary_accuracy', 'sparse_categorical_accuracy'])
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

if __name__ == '__main__':
    main()