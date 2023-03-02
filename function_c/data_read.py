import pandas as pd
from sklearn.preprocessing import LabelEncoder
from function_c.base_util import *
import numpy as np


def read_nametxt(file_name):
    data = []
    file = open(file_name, 'r')
    file_data = file.readlines()
    for row in file_data:
        tmp_list = row.split(': ')
        tmp_list[-1] = tmp_list[-1].replace('.\n', '')
        data.append(tmp_list)
    data = data[1:len(data) + 1]
    data.append(['target', 'symbolic'])
    return data


if 'ab' in 'abc':
    print('ok')
def data_read(data_name):
    if data_name == 'adult':
        data_ = pd.read_csv('data/adult.csv')
        for i in data_.columns:
            data_[i] = LabelEncoder().fit_transform(data_[i].values)
        epoch, batch_size = 60, 300
        target_name = 'income'

    if data_name == 'german':
        data_ = pd.read_csv('./data/german.csv')
        epoch, batch_size = 300, 50
        target_name = 'Y'
        for i in data_.columns:
            data_[i] = LabelEncoder().fit_transform(data_[i].values)

    if data_name == 'taiwan':
        data_ = pd.read_excel('./data/taiwan.xls', sheet_name='Data', header=0)
        data_.drop(0, axis=0, inplace=True)
        data_.drop(['Unnamed: 0'], axis=1, inplace=True)
        data_ = data_.infer_objects()
        target_name = 'Y'
        epoch, batch_size = 60, 300

    if data_name == 'covertype':
        data_ = pd.read_csv('data/' + data_name + '.csv')
        target_name = 'Cover_Type'
        epoch, batch_size = 60, 300
        data_.drop(['Unnamed: 0'], axis=1, inplace=True)
        data_, _ = tail_del(data_, rate=0.98)

    if data_name == 'Intrusion':
        data_name = np.array(read_nametxt('./data/' + data_name + '/kddcup.names'))
        df = pd.read_csv('data/Intrusion/kddcup.data_10_percent.gz', names=data_name[:, 0])
        for k in df.columns:
            if df[k].dtypes == 'object':
                df[k] = LabelEncoder().fit_transform(df[k])
        target_name = 'target'
        epoch, batch_size = 60, 300
        data_, _ = tail_del(df, rate=0.98)

    data = data_.drop([target_name], axis=1, inplace=False)
    targets = data_[target_name]
    return data, targets, target_name, epoch, batch_size


