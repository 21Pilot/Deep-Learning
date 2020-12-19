# -*- coding: utf-8 -*-
# Flora Seo 
# 2020/11/14
# Deep Learning_Project6

# AlexNet

# Set Up
import keras
from keras.models import Sequential 
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np

image_shape = (227,227,3)
np.random.seed(1000)
model = Sequential()

# Convolution Layer_1
# First layer with 96 Filters, input shape of 227* 227* 3
# Kernel size of 11*11, Striding 4*4 with Relu as activation function 
model.add(Conv2D(filters=96, input_shape=image_shape, kernel_size =(11,11), 
                 strides=(4,4), padding='valid'))
model.add(Activation('relu'))

# Max Pooling_1
model.add(MaxPooling2D(pool_size =(3,3), strides=(2,2), padding='valid'))

# Convolution Layer_2
model.add(Conv2D(filters=256, kernel_size =(5,5), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# Max Pooling_2
model.add(MaxPooling2D(pool_size =(3,3), strides=(2,2), padding='valid'))

## Convolution Layer_3
model.add(Conv2D(filters=384, kernel_size =(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# Convolution Layer_4
model.add(Conv2D(filters=384, kernel_size =(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# Convolution Layer_5
model.add(Conv2D(filters=256, kernel_size =(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# Max Pooling_2
model.add(MaxPooling2D(pool_size =(3,3), strides=(2,2), padding='valid'))

# Flatten
model.add(Flatten())

# 1st Fully Connected Layer, 4096 neurons
model.add(Dense(4096, input_shape=(227,227,3,)))
model.add(Activation('relu'))

# Dropout to prevent overfitting
model.add(Dropout(0.4))

# 2nd Fully Connected Layer
model.add(Dense(4096))
model.add(Activation('relu'))

# Add Dropout
model.add(Dropout(0.4))

# Output Layer
model.add(Dense(1000))
model.add(Activation('softmax'))
model.summary
