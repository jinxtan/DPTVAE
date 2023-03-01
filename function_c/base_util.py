import numpy as np
import pandas as pd

def tail_del(df, rate):
    dele_list = []
    len_1 = df.shape[0]
    for col in df.columns:
        if len(df[col].unique()) < 5:
            if df[col].value_counts().max() / len_1 >= rate:
                dele_list.append(col)
                print("{} 变量分布不均衡，该变量被删除".format(col))
    df = df.drop(dele_list, axis=1)
    return df, dele_list

def data_read_(dir):
    data = pd.read_csv(dir)
    if 'Unnamed: 0' in data.columns:
        data.drop(['Unnamed: 0'], axis=1,inplace=True)
    if 'Unnamed: 0.1' in data.columns:
        data.drop(['Unnamed: 0.1'], axis=1,inplace=True)
    return data

def get_target(data_name):
    if data_name == 'bank':
        target_name = 'isDefault'
    if data_name == 'german':
        target_name = 'Y'
    if data_name == 'adult':
        target_name = 'income'
    if data_name == 'covertype':
        target_name = 'Cover_Type'
    if data_name == 'taiwan':
        target_name = 'Y'
    if data_name == 'Intrusion':
        target_name = 'target'
    return target_name


def spilt(data, target_name):
    x = data.drop(columns=target_name, inplace=False)
    y = data[target_name]
    return x, y


def ci(data):
    l = np.mean(data) - 1.96 * np.std(data) / np.sqrt(len(data) - 1)
    u = np.mean(data) + 1.96 * np.std(data) / np.sqrt(len(data) - 1)
    return l, np.mean(data), u

def value_update(data):
    label = pd.read_csv('result/Intrusion/dict.csv')
    dict_ = {}
    for l in range(len(label)):
        dict_.update({label['value'].iloc[l]:label['label'].iloc[l]})
    data['target'] = data.target.map(dict_)
    return data