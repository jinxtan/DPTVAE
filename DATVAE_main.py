# -*- coding: utf-8 -*-
"""
Created on 2022/6/30 17:05
@author: Jinxtan
email: 20110240017@fudan.edu.cn
PyCharm.py
"""
from models.tvae import TVAESynthesizer
from function_c.corr_dist import *
from function_c.result_output import *
from sklearn.model_selection import train_test_split
from function_c.auto_dc import *
from col_defenition import con_col,discrete_columns
from function_c.data_read import *
import warnings

np.random.seed(1337)
warnings.filterwarnings('ignore')

data_ = pd.read_csv('data/german.csv')
for i in data_.columns:
    data_[i] = LabelEncoder().fit_transform(data_[i].values)

epoch, batch_size = 300, 60 # 根据性能差异可调整为 60， 300
label_col = 'Y'
data_name = 'cy' # 成渝地区数据

# Names of the columns that are discrete
x = data_.drop([label_col], axis=1)
y = data_[label_col]
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, stratify=y, random_state=50)
train = train_x.join(train_y)
test = test_x.join(test_y)

con_c = auto_conc(x, con_col)
tvae = TVAESynthesizer(epochs=epoch, batch_size=batch_size, con_c=con_c)
tvae.fit(train, discrete_columns)
samples = tvae.sample(len(train_x))

corr_com(data_origin = train,samples = samples,data_name=data_name)
distribution(train,samples,discrete_columns,label_col,data_name)

fit_predict(train_x,train_y,test_x,test_y)
g_x = samples.drop(columns = label_col,inplace = False)
g_y = samples[label_col]
fit_predict(g_x,g_y,test_x,test_y)
