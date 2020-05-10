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

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline

from sklearn.utils import resample

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

    count_vect, tfidf_transformer, X_train_tfidf = PreprocessData(X)

    X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, Y, test_size=0.2)

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    scoring = {'accuracy': make_scorer(accuracy_score),
               'roc_auc_score': make_scorer(roc_auc_score)}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=50, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1, scoring=scoring, refit='roc_auc_score')
    # Fit the random search model
    rf_random.fit(X_train, y_train)

    rf_clf = rf_random.best_estimator_
    y_pred = rf_clf.predict(X_test)
    y_pred_prob = rf_clf.predict_proba(X_test)
    y_pred_prob = [p[1] for p in y_pred_prob]

    print("Number of mislabeled points out of a total %d points : %d"
          % (X_test.shape[0], (y_test != y_pred).sum()))
    Accuracy = accuracy_score(y_pred, y_test)
    print("Accuracy : %f" % Accuracy)
    AUROC = roc_auc_score(y_pred, y_pred_prob)
    print("Area Under ROC Curve : %f" % AUROC)

    print(classification_report(y_test, y_pred))

    X_unlabeled = read_unlabeled_data()

    # y_pred = ClassifyPerUser(X_unlabeled, count_vect, tfidf_transformer, rf_clf)
    y_pred = ClassifyAll(X_unlabeled, count_vect, tfidf_transformer, rf_clf)

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