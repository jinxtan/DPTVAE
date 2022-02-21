# -*- coding: utf-8 -*-
"""
Created on 2022/2/18 21:17
@author: Jinxtan
email: 20110240017@fudan.edu.cn
PyCharm.py
"""
from __future__ import print_function
import random
from wandb import magic
import seaborn as sns
from visulization import summarize_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,roc_curve,roc_auc_score,accuracy_score
import tensorflow as tf
from index_function import *
import matplotlib.pyplot as plt
from functools import partial
from ACGAN import ACGAN
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import warnings
import pandas as pd
from imblearn.over_sampling import SMOTE

np.random.seed(1337)
warnings.filterwarnings('ignore')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # 按需动态分配显存
session = tf.compat.v1.Session(config=config)

# load images
def load_real_samples(train=True,dir = 'bank'):
	# load dataset
	data_csv = pd.read_csv('./data_prepare/train_' + dir + '.csv')
	data_csv.drop(['Unnamed: 0', 'loan_id', 'user_id'], axis=1, inplace=True)

	data = data_csv.drop(['isDefault'], axis=1, inplace=False).values
	targets = data_csv['isDefault'].values
	scalr = MinMaxScaler(feature_range=(-1, 1))
	data = scalr.fit_transform(data)

	train_x, test_x, train_y, test_y = train_test_split(data, targets, test_size=0.3, stratify=targets, shuffle = True, random_state=10)
	ros = SMOTE(random_state=0, sampling_strategy='auto')
	# train_x, train_y = ros.fit_resample(train_x, train_y)
	if train:
		return [train_x, train_y]

	else:
		return [test_x,test_y]

# select real samples
def generate_real_samples(dataset, n_samples,iteration=0, train=True):
	# split into images and labels
	images, labels = dataset
	# choose random instances
	labels = np.array(labels)
	if train == False:
		ix = np.random.randint(0, images.shape[0], n_samples)
		X, labels = images[ix], labels[ix]

	X, labels = images[iteration*n_samples:(iteration+1)*n_samples], labels[iteration*n_samples:(iteration+1)*n_samples]
	y = np.ones((n_samples,1))
	return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=2):
	# generate points in the latent space
	x_input = np.random.normal(0,1,(latent_dim, n_samples))
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	l1 = np.zeros(int(2*n_samples/5),dtype = 'int32')
	l2 = np.ones(int(n_samples-int(2*n_samples/5)),dtype = 'int32')
	labels = np.concatenate((l1,l2),axis = 0)
	np.random.shuffle(labels)
	#
	# labels = np.random.randint(0, n_classes, n_samples)
	return [z_input, labels]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	y = np.zeros((n_samples, 1))
# 	y = np.ones((n_samples, 1))
	return [images, labels_input], y

######################################### model define #######################################
# size of the latent space
latent_dim = 100
lr = 0.001
# create the discriminator
model = ACGAN()
optimizer = tf.keras.optimizers.Adam(learning_rate = lr,beta_1 = 0.5)
losses = ['binary_crossentropy','sparse_categorical_crossentropy']

discriminator = model.discriminator
discriminator.summary()
# build the generator
generator = model.generator
generator.summary()

noise = tf.keras.layers.Input(shape = (latent_dim,))
label = tf.keras.layers.Input(shape = (1,))
img = generator([noise,label])

discriminator.trainable = False
valid,target_label = discriminator(img)

combined = tf.keras.Model([noise,label],[valid,target_label])
combined.compile(loss = losses,
					  optimizer = optimizer, metrics = ['accuracy',ks_()])

# load image data
generate_test = False
n_epochs=251
n_batch=400
dataset = load_real_samples()
# calculate the number of batches per training epoch
iterations = int(dataset[0].shape[0] / n_batch)
# calculate the size of half a batch of samples
half_batch = int(n_batch / 2)
if generate_test == True:
	f = 8
else:
	f = 6
all_loss = np.zeros((n_epochs,f))
all_acc = np.zeros((n_epochs,f))
all_ks = np.zeros((n_epochs,f))
X_test,y_test = load_real_samples(train = False)
y_test = y_test.reshape(-1,1)

optimal_discriminator = discriminator
optimal_generator = generator
optimal_loss = 10
# manually enumerate epochs
for i in range(n_epochs):
	resultList = random.sample(range(0,dataset[0].shape[0]),dataset[0].shape[0])
	dataset[0] = dataset[0][resultList]
	dataset[1] = dataset[1][resultList]

	all_loss_ = np.zeros((iterations,f))
	all_acc_ = np.zeros((iterations,f))
	all_ks_ = np.zeros((iterations,f))
	for iteration in range(iterations):
		# get randomly selected 'real' samples
		[X_real,labels_real],y_real = generate_real_samples(dataset,n_samples = half_batch,iteration = iteration,
															train = True)
		labels_real = labels_real.reshape(-1,1)
		[X_fake,labels_fake],y_fake = generate_fake_samples(generator,latent_dim,half_batch)
		labels_fake = labels_fake.reshape(-1,1)
		X = np.concatenate((X_real,X_fake),axis = 0)
		aux_Y = np.concatenate((labels_real,labels_fake),axis = 0)
		Y = np.concatenate((y_real,y_fake),axis = 0)

		# update discriminator model weights
		_,all_loss_[iteration][0],all_loss_[iteration][1],all_acc_[iteration][0],all_ks_[iteration][0], \
		all_acc_[iteration][1],all_ks_[iteration][1] = discriminator.train_on_batch(X,[Y,aux_Y])
		# print('real max: %.3f, real mean: %.3f, real min: %.3f'%(np.max(discriminator.predict(X_real)[0]), np.mean(discriminator.predict(X_real)[0]), np.min(discriminator.predict(X_real)[0])))

		# generate 'fake' examples
		[X_real_test,y_real_test],labels_real_test = generate_real_samples(dataset = [X_test,y_test],
																		   n_samples = half_batch,train = False)
		_,all_loss_[iteration][2],all_loss_[iteration][3],all_acc_[iteration][2],all_ks_[iteration][2], \
		all_acc_[iteration][3],all_ks_[iteration][3] = discriminator.test_on_batch(X_real_test,
																				   [labels_real_test,y_real_test])
		# prepare points in latent space as input for the generator

		# create inverted labels for the fake samples
		for k in range(4):
			[z_input,z_labels] = generate_latent_points(latent_dim,n_batch)
			z_labels = z_labels.reshape(-1,1)
			# scalr = MinMaxScaler(feature_range = (0.7, 1.2))
			# y_gan = np.random.randn(n_batch,1)
			# y_gan = scalr.fit_transform(y_gan)
			y_gan = np.ones((n_batch,1))
			# update the generator via the discriminator's error
			_,all_loss_[iteration][4],all_loss_[iteration][5],all_acc_[iteration][4],all_ks_[iteration][4], \
			all_acc_[iteration][5],all_ks_[iteration][5] \
				= combined.train_on_batch([z_input,z_labels],[y_gan,z_labels])
		if generate_test == True:
			[z_input,z_labels] = generate_latent_points(latent_dim,n_batch)
			# create inverted labels for the fake samples
			y_gan = np.ones((n_batch,1))
			_,all_loss_[iteration][6],all_loss_[iteration][7],all_acc_[iteration][6],all_ks_[iteration][6], \
			all_acc_[iteration][7],all_ks_[iteration][7] \
				= combined.test_on_batch([z_input,z_labels],[y_gan,z_labels])
	monitor = 'val_acc'
	train_pred0 = optimal_discriminator.predict(dataset[0])
	test_pred0 = optimal_discriminator.predict(X_test)
	train_pred = discriminator.predict(dataset[0])
	test_pred = discriminator.predict(X_test)

	##########################################################################################
	# optimal_discriminator,optimal_generator = callback(optimal_discriminator,discriminator,optimal_generator, generator,dataset[0],dataset[1],X_test,y_test,monitor='val_acc')
	all_loss[i] = np.mean(all_loss_,axis = 0)
	all_acc[i] = np.mean(all_acc_,axis = 0)
	all_ks[i] = np.mean(all_ks_,axis = 0)
	times = 15
	if (i > times) & (i % times == 0) & (all_acc[i][1] < all_acc[i - times][1]):
		ReduceLROnPlateau(discriminator)
		ReduceLROnPlateau(combined)
		print('The lr for discriminator is: ',discriminator.optimizer.lr)
		print('The lr for generator is ',combined.optimizer.lr)
	if i % 50 == 0:
		discriminator.save("model/adiscriminator%s.hdf5"%i)
		generator.save("model/agenerator%s.hdf5"%i)
		print('model save..........')

	print('>%d, dis_aux_loss[%.3f,%.3f], g_dis_aux_loss[%.3f,%.3f], dis_aux_acc[%.3f,%.3f], '
		  'g_dis_aux_acc[%.3f,%.3f]' % (i + 1,all_loss[i][0],all_loss[i][1],all_loss[i][4],all_loss[i][5],
										all_acc[i][0],all_acc[i][1],all_acc[i][4],all_acc[i][5],))

def print_result(x,y,discriminator, real = True):
	if real == True:
		validity = np.ones(len(x))
	else:
		validity = np.zeros(len(x))
	y_validity, y_label = discriminator.predict(x)
	print('The discriminator accuracy is: \n',accuracy_score(validity,np.round(y_validity)))
	print(classification_report(y,np.argmax(y_label,axis = 1),
								target_names = ['fully paid','default'],
								digits = 4))
	auc = roc_auc_score(y,y_label[:,1])
	fpr,tpr,thresholds = roc_curve(y,y_label[:,1])
	print('KS value: ', max(tpr - fpr))
	print('AUC value: ', auc)

np.random.randn(100)
x_test, y_test = load_real_samples(train = False)
x1 = x_test[:,::-1]
print('The transpose test part result: \n')
print_result(x1,y_test,discriminator, real = True)

print('The test part result: \n')
print_result(x_test,y_test,discriminator, real = True)

print('The generate data result: \n')
[x_fake,labels_fake],y_fake = generate_fake_samples(generator,latent_dim,len(y_test))
print_result(x_fake,labels_fake,discriminator, real = False)

pd.DataFrame(all_loss).to_csv('result/all_loss.csv')
pd.DataFrame(all_ks).to_csv('result/all_ks.csv')
pd.DataFrame(all_acc).to_csv('result/all_acc.csv')

summarize_curve(all_loss,all_ks, all_acc, generate_test)

all_x = np.concatenate((x_fake,X_test),axis=1)
s = pd.DataFrame(all_x).corr()
for i in range(34):
	print(s[i][i+34])

print('相关性：', pd.DataFrame(np.concatenate((x_fake,np.reshape(labels_fake,(-1,1))),axis=1)).corr()[34])

def generated_feature():
	plt.figure(3)
	plt.subplot(1,2,1)
	plt.title('real feature')
	sns.distplot(dataset[0][0])
	plt.subplot(1,2,2)
	plt.title('generated feature')
	sns.distplot(X_fake[0])
	plt.tight_layout()
	plt.show()

# print(len(np.where(discriminator.predict(x1)[0]<0.5)[0]))
#################################### 测试internet数据 ###############################
data_csv = pd.read_csv('./data_prepare/train_internet.csv')
data_csv.drop(['Unnamed: 0', 'loan_id', 'user_id'], axis=1, inplace=True)
data = data_csv.drop(['isDefault'], axis=1, inplace=False).values
targets = data_csv['isDefault'].values
scalr = MinMaxScaler(feature_range=(-1, 1))
data = scalr.fit_transform(data)
y_validity, y_label = discriminator.predict(data)
print_result(data,targets,discriminator, real = True)