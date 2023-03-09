# -*- coding: utf-8 -*-
"""
Created on 2022/6/30 17:05
@author: Jinxtan
email: 20110240017@fudan.edu.cn
PyCharm.py
"""
from models.DATVAE import DATVAESynthesizer
from eval.corr_dist import *
from eval.evaluation import get_utility_metrics, stat_sim, privacy_metrics
from sdv.evaluation import evaluate
from eval.result_output import *
from function_c.auto_dc import *
from sklearn.model_selection import train_test_split
from function_c.data_read import *
import warnings

np.random.seed(1337)
warnings.filterwarnings('ignore')

data_name = 'german'
x, y, label_col, epoch, batch_size = data_read(data_name)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, stratify=y, random_state=50)
train = train_x.join(train_y)
test = test_x.join(test_y)

discrete_columns, con_c = auto_d(train, rate=0.35, data_name=data_name, auto=True)

DATVAE = DATVAESynthesizer(epochs=epoch, batch_size=batch_size, con_c=con_c)
DATVAE.fit(train, discrete_columns)
samples = DATVAE.sample(len(train_x))

distribution(train,samples,discrete_columns,label_col,data_name)

g_x = samples.drop(columns = label_col,inplace = False)
g_y = samples[label_col]

recall, f1, AUC = fit_predict(g_x,g_y,test_x,test_y,data_name)
kl_ = evaluate(samples, train, metrics=['ContinuousKLDivergence'],
                                                aggregate=False)['normalized_score'].values[0]
Rho_D, Rho_SD = corr_d(train,samples)
DNS,NNDR = privacy_metrics(train, samples)
