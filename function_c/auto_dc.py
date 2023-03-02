# -*- coding: utf-8 -*-
"""
Created on 2022/4/30 10:19
@author: Jinxtan
email: 20110240017@fudan.edu.cn
PyCharm.py
"""
import pandas as pd
import numpy as np
from scipy import stats

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

# Estimate the density of observations at this level
def kde_plot(x, key):
    """Create a 1D grid of evaluation points."""
    x = pd.DataFrame(x, columns=[key])
    x[key].dropna(inplace=True)
    kde = stats.gaussian_kde(x[key], bw_method='scott')
    bw_adjust = 1
    kde.set_bandwidth(kde.factor * bw_adjust)
    bw = np.sqrt(kde.covariance.squeeze())
    clip = (None, None)
    clip_lo = -np.inf if clip[0] is None else clip[0]
    clip_hi = +np.inf if clip[1] is None else clip[1]
    cut = 3
    gridmin = max(x[key].min() - bw * cut, clip_lo)
    gridmax = min(x[key].max() + bw * cut, clip_hi)
    gridsize = 200
    support = np.linspace(gridmin, gridmax, gridsize)
    """ obtain support """

    density = kde(support)
    ## check data is log_scale or not
    log_scale = False
    if log_scale:
        support = np.power(10, support)
    densities = {}
    densities[key] = pd.Series(density, index=support)
    return densities

def auto_conc(data, con_c):
    cv_c = [[] for i in range(2)]
    for key in con_c:
        x = data[key]
        densities = kde_plot(x, key)
        s = np.concatenate((densities[key].index.values.reshape(-1, 1),
                            densities[key].values.reshape(-1, 1)), axis=1)
        count_ = 0
        for k in range(1, len(s)-1):
            if (s[k, 1] > s[k - 1, 1]) and (s[k, 1] > s[k + 1, 1]):
                count_ += 1
        cv_c[0].append(key)
        cv_c[1].append(count_)

    for k in range(len(con_c)):
        if int(cv_c[1][k]) <= 5:
            cv_c[1][k] = 5

    con_c = np.concatenate([np.array(cv_c[0]).reshape(-1, 1), np.array(cv_c[1]).reshape(-1, 1)], axis=1)
    return con_c

def auto_d(data, rate, data_name=None, auto=True):
    if auto == True:
        cv_c = [[] for i in range(3)]
        dis_list = []
        for key in data.columns:
            x = data[key]
            # print(key)
            densities = kde_plot(x, key)
            v_c = data[key].value_counts().index.values
            s = np.concatenate((densities[key].index.values.reshape(-1, 1),
                                densities[key].values.reshape(-1, 1)), axis=1)
            count_ = 0
            # if densities[key].values.sum()<0.001:

            for k in range(1, len(s) - 1):
                # print(k)
                if (s[k, 1] > s[k - 1, 1]) and (s[k, 1] > s[k + 1, 1]):
                    count_ += 1
            cv_c[0].append(key)
            cv_c[2].append(count_)
            cv_c[1].append(count_ / len(v_c))
        s = np.where(np.array(cv_c[1]) > rate)[0]
        cv_c = np.array(cv_c)
        dis_list = list(set(dis_list) | set(cv_c[0][s]))
        con_dist = list(set(data.columns) - set(dis_list))
        h = list(data.columns)
        con_c = [[] for i in range(2)]
        for k in range(len(con_dist)):
            l = h.index(con_dist[k])
            con_c[0].append(cv_c[0][l])
            if int(cv_c[2][l]) <= 5:
                cv_c[2][l] = 5
            con_c[1].append(int(cv_c[2][l]))
        con_c = np.concatenate([np.array(con_c[0]).reshape(-1, 1), np.array(con_c[1]).reshape(-1, 1)], axis=1)
    else:
        if data_name == 'german':
            dis_list = ['A1', 'A3', 'A4', 'A6', 'A7', 'A9', 'A10', 'A12', 'A14', 'A15', 'A17', 'A19',
                        'A20', 'Y']
            con_columns = ['A2', 'A5', 'A8', 'A11', 'A13', 'A16', 'A18']  #
            con_c = auto_conc(data, con_columns)
        if data_name == 'adult':
            dis_list = ['workclass', 'education', 'marital-status',
                        'occupation', 'relationship', 'race', 'sex',
                        'native-country', 'income']
            con_columns = list(set(data.columns.tolist()) - set(dis_list))
            con_c = auto_conc(data, con_columns)
        if data_name == 'Intrusion':
            col_name = np.array(read_nametxt('data/Intrusion/kddcup.names'))
            dis_l = list(col_name[np.where(col_name[:, 1] == 'symbolic'), 0][0])
            dis_list = []

            for l in dis_l:
                if l in data.columns:
                    dis_list.append(l)
            con_columns = list(set(data.columns.tolist()) - set(dis_list))
            con_c = auto_conc(data, con_columns)
        if data_name == 'covertype':
            dis_list = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                        'Vertical_Distance_To_Hydrology',
                        'Horizontal_Distance_To_Roadways',
                        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                        'Horizontal_Distance_To_Fire_Points', 'Cover_Type']

            for k in data.columns:
                if 'Soil_Type' in k or 'Wilderness_Area' in k:
                    dis_list.append(k)
            con_columns = list(set(data.columns.tolist()) - set(dis_list))
            con_c = auto_conc(data, con_columns)

    return dis_list, con_c
