# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 20:58:34 2019

@author: SUBIR
"""
# CNN Regression for Multispectral to Hyperspectral Data Transformation
# Citation: Paul, S., & Nagesh Kumar, D. (2020). Transformation of Multispectral Data to Quasi-Hyperspectral 
# Data using Convolutional Neural Network Regression. IEEE Transactions on Geoscience and Remote Sensing (TGRS), 
# Vol. 59, Issue 4, Pages: 3352 - 3368.

# import the necessary packages
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import scipy.io
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense,Dropout,Input,Activation,Conv2D,Flatten
from keras.utils import plot_model

# Ref_L7 is the '.mat' file of the Landsat7 (i.e. multispectral) data
# This can be imported in Python's 'Variable explorer' using 'Import data' option
ref_l7 = Ref_L7
# The Hyperspectral (i.e. Hyperion) data is divided into 3 parts because of its large size and memory issues
# Ref_Hyp1, Ref_Hyp2, and Ref_Hyp3 are the 3 '.mat' files, which can be read in similar way like Landsat7
ref_hyp = np.dstack((Ref_Hyp1, Ref_Hyp2, Ref_Hyp3))
height, width, channel = ref_l7.shape

# creating patches with size 5*5
patch_size = 5
list_l7_patches = []
list_hyp_patches = []
for i in range(2, height-2):
    for j in range(2, width-2):
        uly=i-2
        ulx=j-2
        l7_patch = ref_l7[uly:uly+patch_size, ulx:ulx+patch_size, :]
        hyp_patch = ref_hyp[i, j, :]
        list_l7_patches.append(l7_patch)
        list_hyp_patches.append(hyp_patch)
l7_patches = np.asarray(list_l7_patches)        
hyp_patches = np.asarray(list_hyp_patches)

# removing the patches having any NaN values
hyp_patches1 = pd.DataFrame(hyp_patches)
hyp_patches1 = hyp_patches1.dropna(axis=0)
l7_patches1 = l7_patches[hyp_patches1.index]
rows, channel = hyp_patches1.shape
x = np.empty( (rows,1) )
for i in range(rows):
    x1 = l7_patches1[i,]
    x[i] = np.isnan(x1).any()
x1 = np.where(x==0)
x1 = x1[0]
hyp_patches1 = pd.DataFrame.as_matrix(hyp_patches1)
hyp_patches1 = hyp_patches1[x1]
l7_patches1 = l7_patches1[x1]

# training (50% samples) and testing (50% samples) data creation
l7_train, l7_test, hyp_train, hyp_test = train_test_split(l7_patches1, hyp_patches1, train_size=0.5, test_size=0.5)

# building 2d CNN network for Regression
input_shape = (5, 5, 6)
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(170))
model.add(Dropout(0.5))
model.add(Dense(170))
model.summary()
#rmsp = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss=keras.losses.mean_squared_error, optimizer=adam, metrics=['acc'])
history = model.fit(l7_train, hyp_train, batch_size=512, epochs=50, verbose=1)#, validation_data=(l7_test, hyp_test))
score = model.evaluate(l7_test, hyp_test, verbose=0)
model.save('m5_adam_C.h5')  # saving the model in a HDF5 file 
#plot_model(model, to_file='model4_adam.jpg')
hyp_pred = model.predict(l7_test, batch_size=512) # prediction of Hyperspectral data for the Testing samples

# The same saved model can be used for Prediction of Hyperspectral data from new Multispectral data
model = load_model('E:/PhD IISc/Work(NA)/CNN_Regression/m5_adam_C.h5')
# 'b1_6' is the '.mat' file of new Multispectral data for which Hyperspectral data needed to be generated
hyp_l7_pred = model.predict(b1_6, batch_size=512)
