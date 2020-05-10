import pandas as pd
import numpy as np
import os
import shutil
import scipy

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, make_scorer
from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline

from sklearn.utils import resample

import tensorflow.keras
from tensorflow.keras.metrics import AUC, MeanSquaredError, Accuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

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

# Function to create model, required for KerasClassifier
# def create_model(input_shape=1000, nn1=1000, nn2=500, nn3=200,
#                  act='relu', optimizer='rmsprop', init='glorot_uniform'):
#         # create model
#         model = Sequential()
#         model.add(Dense(nn1, input_dim=input_shape, kernel_initializer=init, activation=act))
#         model.add(Dense(nn2, kernel_initializer=init, activation=act))
#         model.add(Dense(nn3, kernel_initializer=init, activation='sigmoid'))
#         # Compile model
#         model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#         return model

def create_model2(nn1=20, nn2=20, input_shape=1, output_shape=2, lr=1e-1):

    opt = keras.optimizers.Adagrad(learning_rate=lr, clipnorm=5)

    # create model
    model = Sequential()
    model.add(Dense(nn1, input_dim=input_shape))
    model.add(Dense(nn2))
    model.add(Dense(output_shape, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=[AUC(),MeanSquaredError(), Accuracy()])
    return model

def create_model( nl1=1, nl2=1,  nl3=1,
                 nn1=1000, nn2=500, nn3 = 200, lr=0.01, decay=0., l1=0.01, l2=0.01,
                act = 'relu', dropout=0, input_shape=1000, output_shape=20):
    '''This is a model generating function so that we can search over neural net
    parameters and architecture'''

    opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=decay)
    reg = keras.regularizers.l1_l2(l1=l1, l2=l2)

    model = Sequential()

    # for the firt layer we need to specify the input dimensions
    first = True

    for i in range(nl1):
        if first:
            model.add(Dense(nn1, input_dim=input_shape, activation=act, kernel_regularizer=reg))
            first = False
        else:
            model.add(Dense(nn1, activation=act, kernel_regularizer=reg))
        if dropout != 0:
            model.add(Dropout(dropout))

    for i in range(nl2):
        if first:
            model.add(Dense(nn2, input_dim=input_shape, activation=act, kernel_regularizer=reg))
            first = False
        else:
            model.add(Dense(nn2, activation=act, kernel_regularizer=reg))
        if dropout != 0:
            model.add(Dropout(dropout))

    for i in range(nl3):
        if first:
            model.add(Dense(nn3, input_dim=input_shape, activation=act, kernel_regularizer=reg))
            first = False
        else:
            model.add(Dense(nn3, activation=act, kernel_regularizer=reg))
        if dropout != 0:
            model.add(Dropout(dropout))

    model.add(Dense(output_shape, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=[AUC(),MeanSquaredError(), Accuracy()])
    return model

def main():

    input_file = os.path.abspath(os.path.join(label_file_name))
    input_file_df = pd.read_csv(input_file, index_col=0)

    X_all, Y_all = read_data(input_file_df)
    X_labeled, Y_labeled = read_labeled_data(input_file_df)
    X_semi_labeled, Y_semi_labeled = read_semi_labeled_data(input_file_df)

    X = np.append(X_labeled, X_semi_labeled)
    Y = np.append(Y_labeled, Y_semi_labeled)

    # X = X_labeled
    # Y = Y_labeled

    # X, Y = upsample(X, Y)

    # count_vect, tfidf_transformer, X_train_tfidf = PreprocessData(X_all)
    # X_train_tfidf = X_train_tfidf[0:Y_all.shape[0],]

    # count_vect, tfidf_transformer, X_train_tfidf = PreprocessData(X)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X)

    # hash_vect = HashingVectorizer()
    # X_train_counts = hash_vect.fit_transform(X)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    #scaler = StandardScaler(with_mean=False)
    #scaler = RobustScaler(with_centering=False)
    #scaler = MaxAbsScaler()

    #X_train_tfidf = scaler.fit_transform(X_train_tfidf)

    X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, Y, test_size=0.2)

    y_train1 = to_categorical(y_train)

    # create model
    model = KerasClassifier(build_fn=create_model2, verbose=0)

    # grid search epochs, batch size and optimizer
    epochs = [500, 1000, 2000]
    nn1 = [4, 10, 20]
    nn2 = [4, 10, 20]
    lr = [1e-1, 1e-2]

    param_grid2 = dict(epochs=epochs, nn1=nn1, nn2=nn2, lr=lr, input_shape=[X_train.shape[1]], output_shape=[2])

    ''' 
    # learning algorithm parameters
    lr = [1e-2, 1e-3, 1e-4]
    decay = [1e-6, 1e-9, 0]

    # activation
    activation = ['relu', 'sigmoid']

    # numbers of layers
    nl1 = [0, 1, 2, 3]
    nl2 = [0, 1, 2, 3]
    nl3 = [0, 1, 2, 3]

    # neurons in each layer
    nn1 = [300, 700, 1400, 2100, ]
    nn2 = [100, 400, 800]
    nn3 = [50, 150, 300]

    # dropout and regularisation
    dropout = [0, 0.1, 0.2, 0.3]
    l1 = [0, 0.01, 0.003, 0.001, 0.0001]
    l2 = [0, 0.01, 0.003, 0.001, 0.0001]

    # dictionary summary
    param_grid = dict(
        nl1=nl1, nl2=nl2, nl3=nl3, nn1=nn1, nn2=nn2, nn3=nn3,
        act=activation, l1=l1, l2=l2, lr=lr, decay=decay, dropout=dropout,
        input_shape=[X_train.shape[1]],
        #output_shape=[y_train1.shape[1]]
        output_shape=[2]
    )
    '''


    # scoring = {'accuracy': make_scorer(accuracy_score),
    #            'roc_auc_score': make_scorer(roc_auc_score),
    #            'f1_score': make_scorer(f1_score)}

    search = RandomizedSearchCV(estimator=model, param_distributions=param_grid2, n_iter=1, cv=3, verbose=2,
                                    random_state=42, n_jobs=-1
                                # scoring=scoring,
                                # refit='roc_auc_score'
                                )
    # Fit the random search model
    search.fit(X_train, y_train1)

    clf = search.best_estimator_
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)
    y_pred_prob = [p[1] for p in y_pred_prob]

    print("Number of mislabeled points out of a total %d points : %d"
          % (X_test.shape[0], (y_test != y_pred).sum()))
    Accuracy = accuracy_score(y_pred, y_test)
    print("Accuracy : %f" % Accuracy)
    AUROC = roc_auc_score(y_pred, y_pred_prob)
    print("Area Under ROC Curve : %f" % AUROC)

    print(classification_report(y_test, y_pred))

    X_unlabeled = read_unlabeled_data()

    # y_pred = ClassifyPerUser(X_unlabeled, count_vect, tfidf_transformer, clf)
    y_pred = ClassifyAll(X_unlabeled, count_vect, tfidf_transformer, clf)

    WriteOutput(input_file_df, y_pred)


def WriteOutput(input_file_df, y_pred):
    output_df = input_file_df.values

    output_df[unlabeled_users_start:num_of_users, unlabeled_seg_start:num_of_seg] = y_pred

    output_df = pd.DataFrame(data=output_df, index=input_file_df.index, columns=input_file_df.columns, dtype=int)

    output_file = os.path.abspath(os.path.join(output_folder_name, label_file_name))

    if os.path.exists(os.path.abspath(output_folder_name)):
        shutil.rmtree(os.path.abspath(output_folder_name))
    os.makedirs(os.path.abspath(output_folder_name))

    output_df.to_csv(output_file, index=True)


def ClassifyAll(X_unlabeled, count_vect, tfidf_transformer, clf):

    X_new_counts = count_vect.transform(X_unlabeled.flatten())
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    y_pred = clf.predict(X_new_tfidf)

    y_pred = np.reshape(y_pred,X_unlabeled.shape)

    return y_pred


def ClassifyPerUser(X_unlabeled, count_vect, tfidf_transformer, clf):

    y_pred = np.zeros_like(X_unlabeled)

    for user in range(0, X_unlabeled.shape[0]):
        user_seg = X_unlabeled[user]

        X_new_counts = count_vect.transform(user_seg)
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)

        pred_prob = clf.predict_proba(X_new_tfidf)
        top_ten_pred = np.argsort(pred_prob[:, 0])[0:10]

        malicious = np.zeros_like(user_seg, dtype=int)
        malicious[top_ten_pred] = 1

        y_pred[user] = malicious

    return y_pred


def PreprocessData(X):

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X)

    # hash_vect = HashingVectorizer()
    # X_train_counts = hash_vect.fit_transform(X)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    return count_vect, tfidf_transformer, X_train_tfidf


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


def read_unlabeled_data():

    X_unlabeled = np.empty((num_of_unlabeled_users, num_of_seg-labeled_seg_start),dtype=object)

    for user in unlabeled_users:

        user_file = os.path.abspath(os.path.join(data_folder_name, 'User' + str(user)))

        user_df = pd.read_csv(user_file, header=None, names=['Vocalbulary'])

        user_segments = np.empty((1,num_of_seg-labeled_seg_start),dtype=object)

        for seg in unlabeled_segments:
            user_seg_str = user_df['Vocalbulary'][seg * num_of_words_in_seg:(seg + 1) * num_of_words_in_seg].tolist()
            user_segments[:,seg-labeled_seg_start] = ' '.join(user_seg_str)

        X_unlabeled[user - num_of_unlabeled_users] = user_segments

    return X_unlabeled

def read_semi_labeled_data( labels):
    X_semi_labeled = []
    Y_semi_labeled = []

    for user in semi_labeled_users:

        user_file = os.path.abspath(os.path.join(data_folder_name, 'User' + str(user)))

        user_df = pd.read_csv(user_file, header=None, names=['Vocalbulary'])

        for seg in semi_labeled_segments:
            user_seg = user_df['Vocalbulary'][seg * num_of_words_in_seg:(seg + 1) * num_of_words_in_seg].tolist()

            X_semi_labeled.append(' '.join(user_seg))

            Y_semi_labeled.append((int(labels.values[user, seg])))

    return np.array(X_semi_labeled).reshape(-1, 1),  np.array(Y_semi_labeled).reshape(-1, 1)



def read_labeled_data(labels):

    X_labeled = []
    Y_labeled = []

    for user in labeled_users:

        user_file = os.path.abspath(os.path.join(data_folder_name, 'User' + str(user)))

        user_df = pd.read_csv(user_file, header=None, names=['Vocalbulary'])

        for seg in labeled_segments:
            user_seg = user_df['Vocalbulary'][seg * num_of_words_in_seg:(seg + 1) * num_of_words_in_seg].tolist()

            X_labeled.append(' '.join(user_seg))

            Y_labeled.append((int(labels.values[user, seg])))

    return np.array(X_labeled),  np.array(Y_labeled)

def read_data(labels):

    X = []
    Y = []

    for user in all_users:

        user_file = os.path.abspath(os.path.join(data_folder_name, 'User' + str(user)))

        user_df = pd.read_csv(user_file, header=None, names=['Vocalbulary'])

        for seg in all_segments:
            user_seg = user_df['Vocalbulary'][seg * num_of_words_in_seg:(seg + 1) * num_of_words_in_seg].tolist()

            X.append(' '.join(user_seg))

            if not np.isnan(labels.values[user, seg]):
                Y.append((int(labels.values[user, seg])))

    return np.array(X),  np.array(Y)

if __name__ == '__main__':
    main()