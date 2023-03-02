# -*- coding: utf-8 -*-
"""
Created on 2022/6/15 22:18
@author: Jinxtan
email: 20110240017@fudan.edu.cn
PyCharm.py
"""
from sklearn.metrics import f1_score,recall_score, roc_auc_score, r2_score, accuracy_score
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from .corr_dist import *
from .density_com import *


def fit_predict(train_x, train_y, test_x, test_y, data_name):
    clf = DecisionTreeClassifier(max_depth=30, min_samples_split=2, min_samples_leaf=2, random_state=50)
    clf.fit(train_x, train_y, sample_weight=0.3)
    y_pred = clf.predict_proba(test_x)
    y_p = clf.predict(test_x)

    if data_name not in ['adult', 'german', 'taiwan']:
        multi_class = "ovo"
        ave = 'macro'
        recall = recall_score(test_y, y_p, average=ave)
        f1 = f1_score(test_y, y_p, average=ave)

        if len(y_pred.shape) == 1 or y_pred.shape[1] != 5:
            print('fault')
            auc = 0.5
        elif y_pred.shape[1] == 5:
            auc = roc_auc_score(test_y, y_pred, multi_class=multi_class)
    else:
        ave = "binary"
        multi_class = "raise"

        if len(y_pred.shape) == 1:
            y_pred = y_p
        elif y_pred.shape[1] == 1:
            y_pred = y_p
        elif y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
        else:
            multi_class = "ovo"
            ave = 'macro'
        recall = recall_score(test_y, y_p, average=ave)
        f1 = f1_score(test_y, y_p, average=ave)
        auc = roc_auc_score(test_y, y_pred, multi_class=multi_class)

    return recall, f1, auc

def corr_sign_d(corr1,corr2):
    rho_d = corr1 - corr2
    for k in corr1.columns:
        corr1.loc[corr1[k] > 0] = 1
        corr1.loc[corr1[k] < 0] = -1
        corr2.loc[corr2[k] > 0] = 1
        corr2.loc[corr2[k] < 0] = -1
    rho_sd = corr1 - corr2
    return rho_d,rho_sd

def corr_d(train,samples):
    corr_r = train.corr(method='spearman')
    corr_s = samples.corr(method='spearman')

    rho_d,rho_sd = corr_sign_d(corr_r, corr_s)
    Rho_D = []
    Rho_SD = []

    for i in train.columns:
        if len(samples[i].value_counts()) == 1:
            Rho_D.append(0)
            Rho_SD.append(0)

        else:
            Rho_D.append(abs(rho_d[i]).sum())
            Rho_SD.append(abs(rho_sd[i]).sum())
    return np.sum(Rho_D), np.sum(Rho_SD)

