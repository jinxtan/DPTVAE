# -*- coding: utf-8 -*-
"""
Created on 2022/4/26 16:45
@author: Jinxtan
email: 20110240017@fudan.edu.cn
PyCharm.py
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy import stats


def kde_cal(x,key):
	x = pd.DataFrame(x,columns = [key])
	x[key].dropna(inplace = True)
	kde = stats.gaussian_kde(x[key],bw_method = 'scott')
	bw_adjust = 1
	kde.set_bandwidth(kde.factor * bw_adjust)
	bw = np.sqrt(kde.covariance.squeeze())
	clip = (None,None)  ## 设置无限上限和下限
	clip_lo = -np.inf if clip[0] is None else clip[0]
	clip_hi = +np.inf if clip[1] is None else clip[1]
	cut = 3
	gridmin = max(x[key].min() - bw * cut,clip_lo)
	gridmax = min(x[key].max() + bw * cut,clip_hi)
	return kde,gridmin,gridmax


# Estimate the density of observations at this level
def kde_plot(x,g_x,key):
	"""Create a 1D grid of evaluation points."""
	kde,gridmin,gridmax = kde_cal(x,key)
	kde_g,gridmin_g,gridmax_g = kde_cal(g_x,key)
	grid_min = min(gridmin,gridmin_g)
	grid_max = max(gridmax,gridmax_g)
	gridsize = 200
	support = np.linspace(grid_min,grid_max,gridsize)
	""" 获得 support """
	density = kde(support)
	densities = {}
	densities[key] = pd.Series(density,index = support)
	density_g = kde_g(support)
	densities_g = {}
	densities_g[key] = pd.Series(density_g,index = support)
	return np.concatenate((densities[key].index.values.reshape(-1,1),
						   densities[key].values.reshape(-1,1),
						   densities_g[key].values.reshape(-1,1)),axis = 1)


def integral_(x,g_x,key):
	s = kde_plot(x,g_x,key)
	area = 0
	for k in range(1,len(s)):
		h = s[k,0] + s[k - 1,0]
		area += abs(s[k,1] + s[k - 1,1] - s[k,2] - s[k - 1,2]) * h / 2
	return area

# data_name = 'census'
# data = pd.read_csv('data/census.csv').drop(['Unnamed: 0'],axis = 1)
# data_g = pd.read_csv('data/generate_census.csv').drop(['Unnamed: 0'],axis = 1)
# target_name = 'income'
#
# for i in data.columns:
# 	data[i] = LabelEncoder().fit_transform(data[i].values)
# 	data_g[i] = LabelEncoder().fit_transform(data_g[i].values)
# 	print(i,integral_(x = data[i],g_x = data_g[i],key = i))
