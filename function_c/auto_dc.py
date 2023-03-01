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
