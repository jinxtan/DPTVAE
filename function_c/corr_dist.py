# -*- coding: utf-8 -*-
"""
Created on 2022/4/14 17:05
@author: Jinxtan
email: 20110240017@fudan.edu.cn
PyCharm.py
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def corr_com(data_origin,samples,data_name=None):
	## draw correlation comparasion
	fontsize = data_origin.shape[-1] * (-1) + 80
	plt.figure(num = 1,figsize = (40,25),dpi = 80)
	plt.rcParams['font.sans-serif'] = ['Times New Roman']
	plt.rcParams.update({'font.size': data_origin.shape[-1] * (-1) + 50})
	plt.subplot(1,2,1)
	sns.heatmap(data_origin.corr().round(2),annot = True,vmax = 1,vmin = 0,xticklabels = True,
				yticklabels = True,square = True,cmap = "YlGnBu")
	plt.title('Real correlation',fontsize = fontsize)
	plt.xlabel('correlation',fontsize = fontsize)
	plt.ylabel('feature',fontsize = fontsize)
	plt.tight_layout()
	plt.subplot(122)
	sns.heatmap(samples.corr().round(2),annot = True,vmax = 1,vmin = 0,xticklabels = True,
				yticklabels = True,square = True,cmap = "YlGnBu")
	plt.title('Generated correlation',fontsize = fontsize)
	plt.xlabel('correlation',fontsize = fontsize)
	plt.ylabel('feature',fontsize = fontsize)
	plt.tight_layout()
	# plt.savefig(f'result/{data_name}/{data_name}.svg')
	plt.show()


def corr_d_c(corr1,corr2):
	corr_d = corr1 - corr2
	for k in corr1.columns:
		corr1.loc[corr1[k] > 0] = 1
		corr1.loc[corr1[k] < 0] = -1
		corr2.loc[corr2[k] > 0] = 1
		corr2.loc[corr2[k] < 0] = -1
	corr_c = corr1 - corr2
	return corr_d,corr_c


def corr_dc(corr_d,corr_c):
	## draw correlation comparasion
	fontsize = corr_d.shape[-1] * (-1) + 45
	plt.figure(num = 2,figsize = (40,25),dpi = 80)
	plt.rcParams['font.sans-serif'] = ['Times New Roman']
	plt.rcParams.update({'font.size': corr_d.shape[-1] * (-1) + 45})
	plt.subplot(1,2,1)
	sns.heatmap(corr_d,annot = True,vmax = 1,vmin = 0,xticklabels = True,
				yticklabels = True,square = True,cmap = "YlGnBu")
	plt.title('Correlation D-value',fontsize = fontsize)
	plt.xlabel('correlation',fontsize = fontsize)
	plt.ylabel('feature',fontsize = fontsize)
	plt.tight_layout()
	plt.subplot(122)
	sns.heatmap(corr_c,annot = True,vmax = 1,vmin = 0,xticklabels = True,
				yticklabels = True,square = True,cmap = "YlGnBu")
	plt.title('Correlation count',fontsize = fontsize)
	plt.xlabel('correlation',fontsize = fontsize)
	plt.ylabel('feature',fontsize = fontsize)
	plt.tight_layout()

	plt.show()


# Draw Plot
def distribution(data_origin,samples,discrete_columns,target_name,data_name=None):
	i = 0
	k = 4
	s = 1
	fontsize = 30
	for feature in data_origin.columns:
		if feature != target_name:
			# print(feature)
			# f_c = data_origin[feature].value_counts()
			# l = pd.DataFrame(
			# 	data = np.concatenate((f_c.axes[0].values.reshape(-1,1),f_c.values.reshape(-1,1)),axis = 1),
			# 	columns = ['o_index','o_count'])
			# g_c = samples[feature].value_counts()
			# l1 = pd.DataFrame(
			# 	data = np.concatenate((g_c.axes[0].values.reshape(-1,1),g_c.values.reshape(-1,1)),axis = 1),
			# 	columns = ['g_index','g_count'])
			#
			# print(pd.concat([l,l1],axis = 1))
			plt.rcParams['font.sans-serif'] = ['Times New Roman']
			plt.rcParams.update({'font.size': fontsize})
			if i % 6 == 0:
				plt.figure(num = k,figsize = (32,18),dpi = 80)

			plt.subplot(3,4,s)
			s = s + 1
			sns.kdeplot(data_origin.loc[data_origin[target_name] == 0,feature],shade = True,color = "g",
						label = target_name+" = 0",
						alpha = .7)
			sns.kdeplot(data_origin.loc[data_origin[target_name] == 1,feature],shade = True,color = "black",
						label = target_name+" = 1",
						alpha = .7)
			plt.grid(linestyle = '-.')
			plt.legend()
			plt.xticks(fontsize = fontsize)
			plt.yticks(fontsize = fontsize)
			# Decoration
			if feature in discrete_columns:
				plt.title('Real ' + feature + ' (D)',fontsize = fontsize)
			else:
				plt.title('Real ' + feature + ' (C)',fontsize = fontsize)

			plt.tight_layout()
			plt.subplot(3,4,s)
			s = s + 1
			sns.kdeplot(samples.loc[samples[target_name] == 0,feature],shade = True,color = "dodgerblue",
						label = target_name+" = 0",
						alpha = .7)
			sns.kdeplot(samples.loc[samples[target_name] == 1,feature],shade = True,color = "orange",
						label = target_name+" = 1",
						alpha = .7)
			plt.legend()
			plt.xticks(fontsize = fontsize)
			plt.yticks(fontsize = fontsize)

			plt.grid(linestyle = '-.')
			# Decoration
			if feature in discrete_columns:
				plt.title('Generated ' + feature + ' (D)',fontsize = fontsize)
			else:
				plt.title('Generated ' + feature + ' (C)',fontsize = fontsize)
			plt.tight_layout()
			if i > 4:
				if (i + 1) % 6 == 0:
					# plt.savefig(f'result/{data_name}/{feature}.svg')
					plt.show()
					k = k + 1
					s=1
			if i == samples.shape[1] - 2:
				# plt.savefig(f'result/{data_name}/{feature}.svg')
				plt.show()
				s=1
			# print('i: ',i)
			i = i + 1

# Draw Plot
def distribution_(data_origin, discrete_columns,target_name):
	i = 0
	k = 1
	s = 1
	fontsize = 30
	for feature in data_origin.columns:
		if feature != target_name:
			plt.rcParams['font.sans-serif'] = ['Times New Roman']
			plt.rcParams.update({'font.size': 30})
			if i % 12 == 0:
				plt.figure(num = k,figsize = (32,18),dpi = 80)
				s = 1
			plt.subplot(3,4,s)
			for m in data_origin[target_name].value_counts().index:
				sns.kdeplot(data_origin.loc[data_origin[target_name] == m,feature],shade = True,
							label = target_name + ' = ' +str(m),
							alpha = .7)
			plt.grid(linestyle = '-.')
			plt.legend()
			# Decoration
			if feature in discrete_columns:
				plt.title('Real ' + feature + ' (D)',fontsize = fontsize)
			else:
				plt.title('Real ' + feature + ' (C)',fontsize = fontsize)
			s = s + 1
			plt.tight_layout()
			plt.xticks(fontsize = fontsize)
			plt.yticks(fontsize = fontsize)

			if i > 10:
				if (i + 1) % 12 == 0:
					plt.show()
					k = k + 1
			if i == data_origin.shape[1] - 2:
				plt.show()
			print('i: ',i)
			i = i + 1

