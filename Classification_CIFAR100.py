# -*- coding: utf-8 -*-
# Flora Seo 
# 2020/10/8
# Deep Learning_Project4

# CIFAR 100 data taken from : https://www.cs.toronto.edu/~kriz/cifar.html
# File name : cifar-100-python.tar.gz

# Set up 
import tensorflow as tf
import numpy as np 
from sklearn.model_selection import train_test_split  

# Data set up 
Batch_size = 64
Class_number = 100
# Initial Lambda
Lambda = 0.0001 
Epoch = 10

# CIFAR 100 Class
class Data(object): 
  def __init__(self): 
    # The data, shuffled and split between train and test sets
    # Shuffled to prevent any issue due to order
    (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar100.load_data()
    rand_index_1 = np.arange(len(self.x_train))

    np.random.shuffle(rand_index_1)
    rand_index_2 = np.arange(len(self.x_test))
    np.random.shuffle(rand_index_2)

    self.x_train = self.x_train[rand_index_1]
    self.y_train = self.y_train[rand_index_1]

    self.x_test = self.x_test[rand_index_2]
    self.y_test = self.y_test[rand_index_2]

    self.x_train, self.x_ver, self.y_train, self.y_ver = train_test_split(self.x_train, self.y_train, test_size = 5000)

    # Consider the CIFAR-10 and CIFAR-100 datasets which contain 32 32 pixel color images
    self.x_train = self.x_train.reshape(self.x_train.shape[0], 32, 32, 3).astype('float32')/255.0      	
    self.x_ver = self.x_ver.reshape(self.x_ver.shape[0], 32, 32, 3).astype('float32')/255.0      	
    self.x_test = self.x_test.reshape(self.x_test.shape[0], 32, 32, 3).astype('float32')/255.0 

    mean = np.mean(self.x_train,axis=(0,1,2,3))      	
    std = np.std(self.x_train,axis=(0,1,2,3))      	
    self.x_train = (self.x_train-mean)/(std+1e-7)      	
    self.x_test = (self.x_test-mean)/(std+1e-7) 

    # Convert class vectors to binary class matrices.
    self.y_train = tf.keras.utils.to_categorical(self.y_train, Class_number)      	
    self.y_test = tf.keras.utils.to_categorical(self.y_test, Class_number)      	
    self.y_ver = tf.keras.utils.to_categorical(self.y_ver, Class_number) 
    
class Model(tf.Module):
  def __init__(self):
    # Refernce for keras.Sequential() : https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
    # Refernce for keras.layers.Conv2D : https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
    # and : https://www.programcreek.com/python/example/89658/keras.layers.Conv2D 
    self.model = tf.keras.Sequential()
    self.model.add(tf.keras.layers.Conv2D(32,(3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(Lambda), input_shape = data.x_train.shape[1:]))

    # Refernce for convolution and normalization : https://programtalk.com/python-examples-amp/keras.layers.convolutional.Convolution2D/
    # Model defining : https://programmersought.com/article/67111120957/ 
    # Model defining : normalization -> convolution layer -> maxpooling
  def activation(self):
    self.model.add(tf.keras.layers.Activation('elu'))
  def batch_norm(self):
    self.model.add(tf.keras.layers.BatchNormalization())
  def conv2D_layer(self): 
    self.model.add(tf.keras.layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=tf.keras.regularizers.l2(Lambda)))
  def maxPooling2D(self): 
    self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

if __name__ =="__main__":
  # General reference for activation : https://keras.io/ko/getting-started/sequential-model-guide/
  # and : https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
  # Also used for Dropout application
  # Refernce for Dropout: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
  data = Data()
  model = Model()

  model.activation()
  model.batch_norm()

  model.conv2D_layer()
  model.activation()
  model.batch_norm()

  model.maxPooling2D()
  model.model.add(tf.keras.layers.Dropout(0.4))

  model.conv2D_layer()
  model.activation()
  model.batch_norm

  model.conv2D_layer()
  model.activation()
  model.batch_norm()
  model.maxPooling2D()

  model.model.add(tf.keras.layers.Dropout(0.4))
  model.model.add(tf.keras.layers.Flatten())
  model.model.add(tf.keras.layers.Dense(Class_number, activation = 'softmax'))
  model.model.summary()
  
# Datageneration 
# Refernce for ImageDataGenerator : https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

Data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range = 0.1, 
                                                           horizontal_flip=True)
Data_gen.fit(data.x_train)
# Refernce for model.complie : https://keras.io/api/metrics/
# Refernce for optimizers.Adam : https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam

model.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
                     metrics=['accuracy','top_k_categorical_accuracy'])

# Refernce for fit generator : https://www.geeksforgeeks.org/keras-fit-and-keras-fit_generator/
# Refernce for learning rate scheduler : https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler
# and : https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
# arbitrary number of workers? 

model.model.fit(Data_gen.flow(data.x_train, data.y_train, batch_size=Batch_size),                      	
                          use_multiprocessing=True, steps_per_epoch=data.x_train.shape[0] // Batch_size,
                          epochs=Epoch, verbose=1,validation_data=(data.x_ver, data.y_ver), 
                     	callbacks=[tf.keras.callbacks.LearningRateScheduler(learning_schedule)], workers = 4) 

final_result = model.model.evaluate(data.x_test, data.y_test, batch_size=128, verbose=1)  	
# Test loss, overall accuracy and top 5 accuracy	
# Top5 accuracy needs to be more than 80%
print("Test Loss:", final_result[0])  	
print("Test Accuracy:", final_result[1]) 
print("Test Top-5 Accuracy:", final_result[2]) 

