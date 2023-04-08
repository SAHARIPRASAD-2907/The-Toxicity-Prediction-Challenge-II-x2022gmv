#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import numpy as np


train_data = pd.read_csv("./smile_description_train.csv")
test_data = pd.read_csv("./smile_description_test.csv")
# Replacing Null values of train and test data with 0 
train_data = train_data.fillna(0)
test_data = test_data.fillna(0)
# Split into X and y
X = train_data.drop("label",axis=1)
y = train_data["label"]

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

def create_submission(predict,filename):
    sub_file = pd.read_csv("./data/sample_submission.csv")
    sub_file["Predicted"] = predict
    sub_file.to_csv(filename,index=False)
    print(filename," Created")

def f1_score_xg_boost(X,y):
    skf = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
    cv =  skf.get_n_splits(X, y)
    scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}
    cv_results = cross_validate(xgb.XGBClassifier(random_state=42), X, y, cv=cv,scoring=scoring,verbose=3,n_jobs=-1)
    print("F1 score with ",sum(cv_results["test_f1_score"])/5)
    print("Accuracy score with ",sum(cv_results["test_accuracy"])/5)

def get_count_of_ones_and_twos(predict):
    print("Number of predicted ones",np.count_nonzero(predict==1))
    print("Number of predicted twos",np.count_nonzero(predict==2))

f1_score_xg_boost(X,y,0.35)

xgb_c = xgb.XGBClassifier(random_state=42,n_estimators=700,eta=0.35)
xgb_c.fit(X,y)
predict = xgb_c.predict(test_data)
predict_real = label_encoder.inverse_transform(predict)
get_count_of_ones_and_twos(predict_real)
create_submission(predict_real,"submission_2_mar_24.csv")
