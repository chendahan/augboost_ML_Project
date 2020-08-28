#!/usr/bin/env python
# encoding: utf-8
from time import time
from AugBoost import AugBoostClassifier as ABC
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import pandas as pd
import numpy as np
from scipy.stats import uniform,randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from pandas.api.types import is_string_dtype
#Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# TODO : change path according to the location of the datasets files
dir_name = ""

# model params- used for hyperparameter tuning
model_params = {
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': randint(2, 6),
    'n_estimators': randint(10, 200),
    'subsample': uniform(0.6, 0.4)
}


def multiClassStat(augboost_m, X_test, y_test, y_pred):
    """
    calculate stats for multiclass classification
    :param augboost_m: AugBoost classifier
    :param X_test: test data
    :param y_test: test classification
    :param y_pred: predicted classification using augboost_m
    :return: accuracy, TPR, FPR, precision, roc_auc, PR_curve
    """
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='macro')
    y_prob = augboost_m.predict_proba(X_test)
    roc_auc = 0
    # manually calculate - metrics.roc_auc_score multi_class='ovr' isn't available in 0.19 Sklearn version,
    # which is part of the requirements in AugBoost implementation
    for i,cls in zip(range(len(augboost_m.classes_)), augboost_m.classes_):
        y_test_ = list(map(int, [num==cls for num in y_test]))
        try:
            roc_auc += metrics.roc_auc_score(y_test_,y_prob[:, i])
        except ValueError as inst:
            print(inst)
    roc_auc /= len(y_prob[0])
    # calculate TPR & FPR using the confusion_matrix (macro average)
    conf_mat = metrics.confusion_matrix(y_test,y_pred)
    FP = (conf_mat.sum(axis=0) - np.diag(conf_mat)).astype(float)
    FN = (conf_mat.sum(axis=1) - np.diag(conf_mat)).astype(float)
    TP = (np.diag(conf_mat)).astype(float)
    TN = (conf_mat.sum() - (FP + FN + TP)).astype(float)
    TPR = TP/(TP+FN)
    FPR = FN/(TP+FN)
    TPR = TPR.sum()/len(TPR)
    FPR = FPR.sum()/len(FPR)
    # PR_curve - calculate for each class and use average
    PR_curve = 0
    for i,cls in zip(range(len(augboost_m.classes_)), augboost_m.classes_):
        y_test_ = list(map(int, [num==cls for num in y_test]))
        precision_, recall_, thresholds = metrics.precision_recall_curve(y_test_, y_prob[:, i])
        PR_curve += metrics.auc(recall_, precision_)
    PR_curve /= len(y_prob[0])

    return accuracy, TPR, FPR, precision, roc_auc, PR_curve


def binaryStat(augboost_m, X_test, y_test, y_pred):
    """
    calculate stats for binary classification
    :param augboost_m: AugBoost classifier
    :param X_test: test data
    :param y_test: test classification
    :param y_pred: predicted classification using augboost_m
    :return: accuracy, TPR, FPR, precision, roc_auc, PR_curve
    """
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    y_prob = augboost_m.predict_proba(X_test)
    try:
        roc_auc = metrics.roc_auc_score(y_test, y_prob[:,1])
    except ValueError as inst:
        print(inst)
        roc_auc=None
    # calculate TPR & FPR using the confusion_matrix
    conf_mat = metrics.confusion_matrix(y_test, y_pred)
    TN = conf_mat[0][0]
    FP = conf_mat[0][1]
    FN = conf_mat[1][0]
    TP = conf_mat[1][1]
    TPR = 0 if (TP + FN) == 0 else TP / (TP + FN)
    FPR = 0 if (FP + TN) == 0 else FP / (FP + TN)
    _precision, _recall, thresholds = metrics.precision_recall_curve(y_test, y_prob[:,1])
    PR_curve = metrics.auc(_recall, _precision)

    return accuracy, TPR, FPR, precision, roc_auc, PR_curve


def sanity_check(data, folds):
    """
    check if num samples from class is less than num of folds
    :param folds: k, num of folds in k-fold
    :param data: DataFrame, last column is target
    :return: dropped- dropped class , drop_file - need to drop file, cls_count
    """
    data_count_target = data[data.columns[-1]].value_counts()
    # remove classes with less than 10 lines
    drop_file = False
    cls_count = 0
    dropped = False
    for cls, cnt in data_count_target.iteritems():
        cls_count += 1
        # if num samples from class is less than num of folds, we drop this class
        if cnt < folds:
            data.drop(data[data[data.columns[-1]] == cls].index, inplace=True)
            cls_count -= 1
            dropped = True
    #
    if cls_count < 2:
        drop_file = True
    return dropped, drop_file, cls_count


def pre_process(cls_count, dropped, data):
    """
    :param cls_count: cls_count - class count as calculated in sanity_check
    :param dropped: dropped from sanity_check
    :param data: DataFrame, last column is target
    """
    # convert to 0 1 labels
    if cls_count == 2 and dropped:
        max_val = data[data.columns[-1]].value_counts().index[0]
        data[data.columns[-1]] = data[data.columns[-1]].apply(lambda x: 0 if x == max_val else 1)
    # strings - convert using LabelEncoder
    for i in data.columns:
        if is_string_dtype(data[i]):
            enc = LabelEncoder()
            data[i] = enc.fit_transform(data[i].astype(str))
    # fill nan values
    data.fillna(0, inplace=True)


# create DataFrame and res file for the results
df_results = pd.DataFrame(columns=['dataset_name', 'algorithm_name', 'cross_validation', 'hyper_params',
                          'accuracy', 'TPR', 'FPR', 'precision', 'roc_auc', 'PR_curve', 'training_time', 'inference_time'])
res_file_name = 'results.csv'
df_results.to_csv(res_file_name)
num_folds = 10 # split data into 10-fold
for filename in os.listdir(dir_name):
    print(filename) # debug
    data = pd.read_csv(dir_name + '\\' + filename)
    dropped, drop_file, cls_count = sanity_check(data,num_folds)
    if drop_file:
        print("dropping file: ", filename)
        continue # continue to next file
    pre_process(cls_count, dropped, data)
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    skf = StratifiedKFold(n_splits=num_folds)
    fold=0
    for train_index, test_index in skf.split(X, y):
        fold += 1
        print("fold: ", fold)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # create classifier model
        augboost_model = ABC(augmentation_method='pca')
        # hyperparameter tuning using RandomizedSearchCV
        rs_ = RandomizedSearchCV(augboost_model, model_params, n_iter=50, cv=StratifiedKFold(n_splits=3), random_state=0).fit(X_train, y_train) #50 model, 3 fold
        augboost_model.set_params(**rs_.best_params_)
        # fit model again to get training time
        start = time()
        augboost_model.fit(X_train, y_train)
        training_time = time() - start
        # get inference time
        start = time()
        y_pred = augboost_model.predict(X_test)
        inference_time = (time() - start) / len(X_test) # single inference time
        inference_time *= 1000 # 1000 lines inference time
        if len(y_test.unique()) == 2:
            accuracy, TPR, FPR, precision, roc_auc, PR_curve = binaryStat(augboost_model, X_test, y_test, y_pred)
        else:
            accuracy, TPR, FPR, precision, roc_auc, PR_curve = multiClassStat(augboost_model, X_test, y_test, y_pred)
        df_results = df_results.append({'dataset_name': filename , 'algorithm_name':'AugBoost',
                                        'cross_validation': fold, 'hyper_params': rs_.best_params_,
                                        'accuracy': accuracy, 'TPR': TPR, 'FPR': FPR, 'precision': precision,
                                        'roc_auc':roc_auc, 'PR_curve':PR_curve, 'training_time':training_time,
                                        'inference_time':inference_time}, ignore_index=True)
    # write df_results to file
    df_results.to_csv(res_file_name, mode='a', header=False)
