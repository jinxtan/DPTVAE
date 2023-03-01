# -*- coding: utf-8 -*-
"""
Created on 2022/4/14 11:08
@author: Jinxtan
email: 20110240017@fudan.edu.cn
PyCharm.py
"""
from collections import namedtuple
import pandas as pd
import numpy as np
import torch
from col_defenition import con_col_swish
from rdt.transformers.numerical import BayesGMMTransformer
from rdt.transformers.categorical import OneHotEncodingTransformer

from models.base import random_state

SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo', [
        'column_name', 'column_type', 'transform', 'output_info', 'output_dimensions'
    ]
)

class DataTransformer():
	def __init__(self,max_clusters=5,weight_threshold=0.005):
		"""Create a data transformer.

		Args:
			max_clusters (int):
				Maximum number of Gaussian distributions in Bayesian GMM.
			weight_threshold (float):
				Weight threshold for a Gaussian distribution to be kept.
		"""
		self._max_clusters = max_clusters
		self._weight_threshold = weight_threshold

	def _validate_discrete_columns(self,train_data,discrete_columns):
		"""Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
		if isinstance(train_data,pd.DataFrame):
			invalid_columns = set(discrete_columns) - set(train_data.columns)
		elif isinstance(train_data,np.ndarray):
			invalid_columns = []
			for column in discrete_columns:
				if column < 0 or column >= train_data.shape[1]:
					invalid_columns.append(column)
		else:
			raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

		if invalid_columns:
			raise ValueError(f'Invalid columns found: {invalid_columns}')

	def _fit_continuous(self,data,max_cluster):
		"""Train Bayesian GMM for continuous columns.

		Args:
			data (pd.DataFrame):
				A dataframe containing a column.

		Returns:
			namedtuple:
				A ``ColumnTransformInfo`` object.
		"""
		column_name = data.columns[0]
		index = np.where(max_cluster[:,0]==column_name)[0][0]
		gm = BayesGMMTransformer(max_clusters=int(max_cluster[index,1]))
		gm.fit(data,[column_name])
		num_components = sum(gm.valid_component_indicator)

		if column_name in con_col_swish:
			return ColumnTransformInfo(
				column_name = column_name,column_type = 'continuous',transform = gm,
				output_info = [SpanInfo(1,'swish'),SpanInfo(num_components,'softmax')],
				output_dimensions = 1 + num_components)
		else:
			return ColumnTransformInfo(
				column_name=column_name, column_type='continuous', transform=gm,
				output_info=[SpanInfo(1, 'relu'), SpanInfo(num_components, 'softmax')],
				output_dimensions=1 + num_components)

	def _fit_discrete(self,data):
		"""Fit one hot encoder for discrete column.

		Args:
			data (pd.DataFrame):
				A dataframe containing a column.

		Returns:
			namedtuple:
				A ``ColumnTransformInfo`` object.
		"""
		column_name = data.columns[0]
		ohe = OneHotEncodingTransformer()
		ohe.fit(data,[column_name])
		num_categories = len(ohe.dummies)

		return ColumnTransformInfo(
			column_name = column_name,column_type = 'discrete',transform = ohe,
			output_info = [SpanInfo(num_categories,'softmax')],
			output_dimensions = num_categories)

	def _transform_continuous(self,column_transform_info,data):
		column_name = data.columns[0]
		data[column_name] = data[column_name].to_numpy().flatten()
		gm = column_transform_info.transform
		transformed = gm.transform(data,[column_name])

		#  Converts the transformed data to the appropriate output format.
		#  The first column (ending in '.normalized') stays the same,
		#  but the lable encoded column (ending in '.component') is one hot encoded.
		output = np.zeros((len(transformed),column_transform_info.output_dimensions))
		output[:,0] = transformed[f'{column_name}.normalized'].to_numpy()
		index = transformed[f'{column_name}.component'].to_numpy().astype(int)
		output[np.arange(index.size),index + 1] = 1.0

		return output

	def _transform_discrete(self,column_transform_info,data):
		ohe = column_transform_info.transform
		return ohe.transform(data).to_numpy()

	def fit(self,train_data,discrete_columns=()):
		"""Fit the raw data to the discrete and continuous  formats.

		Args:
			train_data (numpy.ndarray or pandas.DataFrame):
				Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
			discrete_columns (list-like):
				List of discrete columns to be used to generate the Conditional
				Vector. If ``train_data`` is a Numpy array, this list should
				contain the integer indices of the columns. Otherwise, if it is
				a ``pandas.DataFrame``, this list should contain the column names.
		"""
		self._validate_discrete_columns(train_data,discrete_columns)

		self.output_info_list = []
		self.output_dimensions = 0
		self.dataframe = True

		if not isinstance(train_data,pd.DataFrame):
			self.dataframe = False
			# work around for RDT issue #328 Fitting with numerical column names fails
			discrete_columns = [str(column) for column in discrete_columns]
			column_names = [str(num) for num in range(train_data.shape[1])]
			train_data = pd.DataFrame(train_data,columns = column_names)

		self._column_raw_dtypes = train_data.infer_objects().dtypes
		self._column_transform_info_list = []
		for column_name in train_data.columns:
			if column_name in discrete_columns:
				column_transform_info = self._fit_discrete(train_data[[column_name]])
			else:
				column_transform_info = self._fit_continuous(train_data[[column_name]],max_cluster=self._max_clusters)

			self.output_info_list.append(column_transform_info.output_info)
			self.output_dimensions += column_transform_info.output_dimensions
			self._column_transform_info_list.append(column_transform_info)

	def transform(self,train_data):
		"""Take raw data and output a matrix data."""
		if not isinstance(train_data,pd.DataFrame):
			column_names = [str(num) for num in range(train_data.shape[1])]
			train_data = pd.DataFrame(train_data,columns = column_names)

		column_data_list = []
		for column_transform_info in self._column_transform_info_list:
			column_name = column_transform_info.column_name
			data = train_data[[column_name]]
			if column_transform_info.column_type == 'continuous':
				column_data_list.append(self._transform_continuous(column_transform_info,data))
			else:
				column_data_list.append(self._transform_discrete(column_transform_info,data))
		train_data = np.concatenate(column_data_list, axis=1).astype(float)
		return train_data

	def _inverse_transform_continuous(self,column_transform_info,column_data,sigmas,st):
		gm = column_transform_info.transform
		data = pd.DataFrame(column_data[:,:2],columns = list(gm.get_output_types()))
		data.iloc[:,1] = np.argmax(column_data[:,1:],axis = 1)
		if sigmas is not None:
			selected_normalized_value = np.random.normal(data.iloc[:,0],sigmas[st])
			data.iloc[:,0] = selected_normalized_value

		return gm.reverse_transform(data,[column_transform_info.column_name])

	def _inverse_transform_discrete(self,column_transform_info,column_data):
		ohe = column_transform_info.transform
		data = pd.DataFrame(column_data,columns = list(ohe.get_output_types()))
		return ohe.reverse_transform(data)[column_transform_info.column_name]

	def inverse_transform(self,data,sigmas=None):
		"""Take matrix data and output raw data.

		Output uses the same type as input to the transform function.
		Either np array or pd dataframe.
		"""
		st = 0
		recovered_column_data_list = []
		column_names = []
		for column_transform_info in self._column_transform_info_list:
			dim = column_transform_info.output_dimensions
			column_data = data[:,st:st + dim]
			if column_transform_info.column_type == 'continuous':
				recovered_column_data = self._inverse_transform_continuous(
					column_transform_info,column_data,sigmas,st)
			else:
				recovered_column_data = self._inverse_transform_discrete(
					column_transform_info,column_data)

			recovered_column_data_list.append(recovered_column_data)
			column_names.append(column_transform_info.column_name)
			st += dim

		recovered_data = np.column_stack(recovered_column_data_list)
		recovered_data = (pd.DataFrame(recovered_data,columns = column_names)
						  .astype(self._column_raw_dtypes))
		if not self.dataframe:
			recovered_data = recovered_data.to_numpy()

		return recovered_data

	def convert_column_name_value_to_id(self,column_name,value):
		"""Get the ids of the given `column_name`."""
		discrete_counter = 0
		column_id = 0
		for column_transform_info in self._column_transform_info_list:
			if column_transform_info.column_name == column_name:
				break
			if column_transform_info.column_type == 'discrete':
				discrete_counter += 1

			column_id += 1

		else:
			raise ValueError(f"The column_name `{column_name}` doesn't exist in the data.")

		ohe = column_transform_info.transform
		data = pd.DataFrame([value],columns = [column_transform_info.column_name])
		one_hot = ohe.transform(data).to_numpy()[0]
		if sum(one_hot) == 0:
			raise ValueError(f"The value `{value}` doesn't exist in the column `{column_name}`.")

		return {
			'discrete_column_id': discrete_counter,
			'column_id': column_id,
			'value_id': np.argmax(one_hot)
		}