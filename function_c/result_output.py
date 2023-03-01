# -*- coding: utf-8 -*-
"""
Created on 2022/6/15 22:18
@author: Jinxtan
email: 20110240017@fudan.edu.cn
PyCharm.py
"""
from sklearn.metrics import classification_report,roc_curve,roc_auc_score,r2_score,accuracy_score
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from .corr_dist import *
from .density_com import *

def fit_predict(X_train,y_train,X_test,y_test):
	clf = DecisionTreeClassifier(max_depth = 30, min_samples_split = 2, min_samples_leaf = 2, random_state = 50)
	clf.fit(X_train,y_train,sample_weight = 0.3)
	print('class distribution: ',y_train.value_counts())
	print(classification_report(clf.predict(X_test),y_test,
								digits = 4))
	if clf.predict_proba(X_test).shape[1]>2:
		print('AUC value: ',roc_auc_score(y_test,clf.predict_proba(X_test),multi_class='ovo'))
	if (clf.predict_proba(X_test).shape[1]>1) & (clf.predict_proba(X_test).shape[1]<=2):
		print('AUC value: ',roc_auc_score(y_test,clf.predict_proba(X_test)[:,1]))
		fpr,tpr,thresholds = roc_curve(y_test,clf.predict_proba(X_test)[:,1])
		print('KS value: ',max(tpr - fpr))
		print('R2 score: ',r2_score(y_true = y_test,y_pred = clf.predict_proba(X_test)[:,1]))
