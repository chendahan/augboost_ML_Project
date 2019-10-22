#!/usr/bin/env python
# encoding: utf-8

from AugBoost import AugBoostClassifier as ABC
from sklearn.model_selection import train_test_split
from sklearn import datasets

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

abc = ABC(n_estimators=10, max_epochs=10, learning_rate=0.1, n_features_per_subset=round(X_train.shape[0] / 3),
          trees_between_feature_update=10, augmentation_method='nn', save_mid_experiment_accuracy_results=False)

abc.fit(X = X_train, y = y_train)
res = abc.predict(X_test)
score = abc.score(X_test,y_test)
print(score)

