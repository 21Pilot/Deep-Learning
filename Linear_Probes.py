# -*- coding: utf-8 -*-
# Flora Seo 
# 2020/10/30
# Deep Learning_Project5

# Set Up
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model, Sequential, layers
from tensorflow.keras.layers import Conv2D, Flatten, Dense, ReLU, MaxPool2D, Softmax, Dropout 

# Variable Set Up 
Label = 10 # Number of labels
Epoch = 10  # Number of epochs(self iteration)
Img_Dim = 28  # Image dimension
Channel = 1  # Number of channels
Batch = 512 # Batch size 

# Reference 2
# Each Probe's title is listed 
probe_name = ["input", "conv1_preact", "conv1_postact", "conv1_postpool", "conv2_preact", 
         "conv2_postact", "conv2_postpool", "fc1_preact", "fc1_postact", "logits"]
         
# MNIST data load , with train and test set.
# MNIST convolutional model provided by the tensorflow/models/image/mnist/convolutional.py
MNIST = tf.keras.datasets.mnist
(X_train, y_train),(X_test, y_test) = MNIST.load_data()

# Use only a random subset of the features, but always the same ones
# Randomize Data, 5000/10000

randomize = np.arange(10000)
np.random.shuffle(randomize)
X_test = X_test[randomize]
y_test = y_test[randomize]

randomize = np.arange(50000)
np.random.shuffle(randomize)
X_train = X_train [randomize]
y_train = y_train [randomize]

#shape test data
X_test = X_test.reshape(X_test.shape[0], Img_Dim, Img_Dim, Channel)
X_test = X_test.astype('float32')
X_test /= 255.0
#y_test = tf.keras.utils.to_categorical(y_test, Label)

#shape training data
X_train = X_train.reshape(X_train.shape[0],Img_Dim,Img_Dim, Channel)
X_train = X_train.astype('float32')
X_train/= 255.0
#y_train = tf.keras.utils.to_categorical(y_train, Label)

X_test = X_test.reshape(X_test.shape[0], Img_Dim, Img_Dim,  Channel).astype('float32')/255.0 
X_train = X_train.reshape(X_train.shape[0], Img_Dim, Img_Dim,  Channel).astype('float32')/255.0

# Linear Classifier Set Up 
class linearClassifier(layers.Layer): 
  def __init__(self):
    super(linearClassifier,self).__init__() 
    self.dense = Dense(Label)
    self.flatten = Flatten()

  # Call
  def call(self,x):
    return self.dense(self.flatten(x))
    
 # Nerual Network model with new probe insertsion on each layer 
class NN(Model):
  def __init__(self):
    super(NN, self).__init__()
    self.my_layers = []
    self.my_probes = {}

    # Separate the denotation of probe number and according layer number 
    # Layer denoted will be one less than the # of inserted probe 
    self.layer_num = -1

    # Start the NN cycle with inserting 0 probe 
    self.probe_insert(0)

    #CONV 5by5 32 filters - probe 1 addition 
    self.my_layers.append(Conv2D(32, [5, 5], strides=(1, 1), padding='same')) 
    self.probe_insert(1)

    #ReLU - probe 2 addition 
    self.my_layers.append(ReLU()) 
    self.probe_insert(2)

    #MAXPOOL 2by2 - probe 3 addition 
    self.my_layers.append(MaxPool2D(pool_size=(2, 2), padding='same')) 
    self.probe_insert(3)

    #CONV 5by5 64 filters - probe 4 addition 
    self.my_layers.append(Conv2D(64, [5, 5], strides=(1, 1), padding='same')) 
    self.probe_insert(4)

    #ReLU - probe 5 addition 
    self.my_layers.append(ReLU()) 
    self.probe_insert(5)

    #MAXPOOL 2by2 - probe 6 addition 
    self.my_layers.append(MaxPool2D(pool_size=(2, 2), padding='same')) 
    self.probe_insert(6)

    #probe 7 addition
    self.my_layers.append(Flatten()) 
    self.my_layers.append(Dense(512, kernel_regularizer=l2(5e-4), bias_regularizer=l2(5e-4)))
    self.probe_insert(7)

    #ReLU - probe 8 addition 
    self.my_layers.append(ReLU()) 
    self.dense = Dropout(0.5)
    self.probe_insert(8)

    #probe 9 addition
    self.my_layers.append(Dense(Label, kernel_regularizer=l2(5e-4), bias_regularizer=l2(5e-4)))
    self.probe_insert(9)

  # probe insertion
  def probe_insert(self, key): 
      self.my_probes[key] = linearClassifier()

# Blocking the backpropagation from the probes to the model itself can be achieved by
# using tf.stop_gradient in Tensorflow 
# Call 
  def call(self, x):
    probe = self.my_probes[0]
    if self.layer_num == -1: # for network training 
      for (i, layer) in enumerate(self.my_layers):
        if i == 10:
          x = self.dense(x)
          x = layer(x)
          return x       
    else:
      for layer in self.my_layers[0:self.layer_num]: 
        x = layer(x)
        x = tf.stop_gradient(x)
        probe = self.my_probes[self.layer_num]
      return probe(x)

model = NN()   

def probe_training(weights):
  # Initiate probe error array 
  probe_error = []

  for layer_num in model.my_probes.keys():
    model.reset_metrics()
    model.layer_num = layer_num
    model.compile(optimizer = probe_optimizer, loss = probe_loss, metrics = None, 
                  loss_weights = None, weighted_metrics = None)
    model.set_weights(weights)

    model.fit(x = X_train, y = y_train, batch_size = Batch, epochs = Epoch, verbose = 2,
             validation_split=0.2)
    
    # Test loss and accurcay evaluated
    test_loss, test_accuracy = model.evaluate(x=X_test, y = y_test, batch_size = Batch, verbose=2, steps=None)
    # test_loss,test_accuracy = model.evaluate(X_test,y_test,verbose = 2)
    # Probe error is evaluated in respect to test accuracy 
    probe_error.append(1-test_accuracy)

    return probe_error

# learning rate, decay rate, momentum defined
L_rate = 0.01
Decay_rate = 0.95
Momentum = 0.95

# LR Scehedule/ Optimizer/ Loss 
# Learning rate schedule w/ exponential decay
learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay (initial_learning_rate = L_rate, 
                                                                         decay_steps = len(X_train), 
                                                                         decay_rate = Decay_rate, staircase = True, name ='ED')
# Optimizer 
probe_optimizer = tf.optimizers.SGD(learning_rate = learning_rate_schedule, momentum = Momentum , 
                                      nesterov= False,name = 'SGD')
# Loss : Train linear classifier using softmax cross-entrophy loss with SparseCategorical Cross entropy 
probe_loss = tf.losses.SparseCategoricalCrossentropy(from_logits = True, 
                                                      reduction = 'auto', name = 'SCC')

model.compile(optimizer = probe_optimizer, loss = probe_loss, metrics=['accuracy'],weighted_metrics=None)
for layer_num in model.my_probes.keys():
  model.layer_num = layer_num
  model(X_train[0:Batch])
  weights = model.get_weights() 


# train the probes with pre-trained weights
probe_error = probe_training(weights) 
plt.figure(figsize=(20,10))
index = range(len(probe_name)) 
plt.plot(index, probe_error)
plt.xticks(index, probe_name, rotation=20) 
axes = plt.gca()
axes.set(ylim=(0,0.1)) 
plt.ylabel("test prediction error") 
plt.title("Figure 5a")
plt.show()

#train model and save weights
model.layer_num = -1
model.compile(optimizer=optimizer, loss=probe_loss, metrics=['accuracy']) 
model.fit(X_train, y_train, batch_size=Batch, epochs=Epoch, verbose=2) 
weights = model.get_weights()


# train the probes with post-trained weights 
probe_error_trained = probe_training(weights) 
plt.figure(figsize=(20,10))
index = range(len(probe_name)) 
plt.plot(index, probe_error_trained) 
plt.xticks(index, probe_name, rotation=20) 
axes = plt.gca()
axes.set(ylim=(0,0.1)) 
plt.ylabel("test prediction error") 
plt.title("Figure 5b")
plt.show()

def probe_training(weights): 
  probe_error = []

  for layer_num in model.my_probes.keys():
    model.reset_metrics()
    model.layer_num = layer_num

model.compile(optimizer=optimizer, loss= probe_loss, metrics=['accuracy'])
weights = model.get_weights()
model.set_weights(weights)

model.fit(x= X_train, y= y_train, batch_size= Batch, epochs= Epoch, verbose=2,validation_split=1/6)

# Test loss and accurcay evaluated
test_loss,test_accuracy = model.evaluate(x=X_test, y= y_test, batch_size=None, verbose=2, 
                                         sample_weight=None, steps=None, callbacks=None, 
                                         max_queue_size=10, workers=1, 
                                         use_multiprocessing=False, return_dict=False)

# test_loss,test_accuracy = model.evaluate(X_test,y_test,verbose = 2)
# Probe error is evaluated in respect to test accuracy 
probe_error.append(1-test_accuracy)

return probe_error 

# learning rate, decay rate, momentum defined
L_rate = 0.01
Decay_rate = 0.95
Momentum = 0.95

# LR Scehedule/ Optimizer/ Loss 
# Learning rate schedule w/ exponential decay
learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay (initial_learning_rate = L_rate, 
                                                                         decay_steps = len(X_train), 
                                                                         decay_rate = Decay_rate, staircase = True, name ='ED')

# Optimizer 
probe_optimizer = tf.optimizers.SGD(learning_rate = learning_rate_schedule, momentum = Momentum, 
                                      nesterov= False,name = 'SGD')
# Loss : Train linear classifier using softmax cross-entrophy loss with SparseCategorical Cross entropy 
probe_loss = tf.losses.SparseCategoricalCrossentropy(from_logits = True, 
                                                      reduction = 'auto', name = 'SCC')

# Complied, fit, and trained/tested
model.compile(optimizer=optimizer, loss= probe_loss, metrics=['accuracy'])
for layer_num in model.my_probes.keys():
  model.layer_num = layer_num
  model(X_train[0:Batch])
weights = model.get_weights()


# train the probes with pre-trained weights
probe_error = probe_training(weights) 
plt.figure(figsize=(20,10))
index = range(len(probe_name)) 
plt.plot(index, probe_error)
plt.xticks(index, probe_name, rotation=20) 
axes = plt.gca()
axes.set(ylim=(0,0.1)) 
plt.ylabel("test prediction error") 
plt.title("Figure 5a")
plt.show()

#train model and save weights
model.layer_num = -1
model.compile(optimizer=optimizer, loss=probe_loss, metrics=['accuracy']) 
model.fit(X_train, y_train, batch_size=Batch, epochs=Epoch, verbose=2) 
weights = model.get_weights()


# train the probes with post-trained weights 
probe_error_trained = probe_training(weights) 
plt.figure(figsize=(20,10))
index = range(len(probe_name)) 
plt.plot(index, probe_error_trained) 
plt.xticks(index, probe_name, rotation=20) 
axes = plt.gca()
axes.set(ylim=(0,0.1)) 
plt.ylabel("test prediction error") 
plt.title("Figure 5b")
plt.show()
