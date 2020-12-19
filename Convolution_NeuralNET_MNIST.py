# -*- coding: utf-8 -*-
# Flora Seo 
# 2020/9/30
# Frequentist Machine Learning_Project3

# Data from : http://yann.lecun.com/exdb/mnist// the MNIST database of handwritten digits 
# Training: 'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'
# Test : 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'

# Set up 
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import pydot 
from tqdm import trange 
import gzip 

# Data set up 
Batch_size = 50
Filter_number = 36 

# New concepts : Kernel/ Epochs 
# Kernel : class of algorithms for pattern analysis 
Kernel_size = 10 
# Epochs : number of passes of the entire training dataset the learning algorithm has completed.
# any number of epochs over 2 takes a while to execute..
Epoch_number  = 5 

# Class for reading train/ test images/labels and to extract MNIST from data 
# Data from : http://yann.lecun.com/exdb/mnist// the MNIST database of handwritten digits 

class Data(object):
  def __init__(self):
    # Training data
    self.train_images, self.train_labels = self.MNIST_from_data('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
    # features (image)         
    features, labels  = self.train_images, self.train_labels
    # Randomize input for 50000 training images + 10000 test images
    rand_index = np.arange(60000)        
    np.random.shuffle(rand_index)    

    # 50000 training images/labels (train_feature) / 10000 test iamges/labels (validation_feature)        
    # Reference for tf.keras.utils.to_categorical : https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
    # tf.keras.utils.to_categorical: Converts a class vector (integers) to binary class matrix.
    rand_features = np.reshape(features[rand_index], [-1,28,28,1])      
    self.train_feature, self.valid_feature = rand_features[:-10000], rand_features[10000:]

    rand_labels = tf.keras.utils.to_categorical(labels[rand_index], 10) 
    self.train_label, self.valid_label = rand_labels[:-10000], rand_labels[10000:] 

    # Test data 
    self.test_images, self.test_labels = self.MNIST_from_data('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')         
    self.test_images = np.reshape(self.test_images, [-1,28,28,1]) 
    self.test_labels = tf.keras.utils.to_categorical(self.test_labels, 10) 

  def MNIST_from_data (self, filename_images, filename_labels):
    # Reference for gzip.open : https://www.tutorialspoint.com/python-support-for-gzip-files-gzip 
    # Images
    with gzip.open(filename_images, 'rb') as f: 
      # Reference for np.frombuffer : https://d-tail.tistory.com/32 
      #  np.frombuffer : get array from raw data
      image_data = np.frombuffer(f.read(), np.uint8, offset=16)
      # 28 by 28 pixel handwritten digit zero through nine
      image_data = image_data.reshape ( -1,1,28,28)/255.0
      # Labels 
      with gzip.open(filename_labels, 'rb') as f: 
        label_data = np.frombuffer(f.read(),  np.uint8, offset=8)
    return image_data, label_data 
    
class Model(tf.Module):     
  def __init__(self): 
    # https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a\
    # https://faroit.com/keras-docs/1.0.6/getting-started/sequential-model-guide/

    # Reference for tf.keras.Sequential : https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
    # tf.keras.Sequential : groups a linear stack of layers into a tf.keras.Model.
    self.model = tf.keras.Sequential() 

    # Reference for tf.keras.layers : Conv2D https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
    # tf.keras.layers : spatial convolution over  (Optionally convolutional nueural network) 
    # tf.keras.layers input variable : https://keras.io/api/layers/convolution_layers/convolution2d/      
    self.model.add(tf.keras.layers.Conv2D(filters= Filter_number, kernel_size= Kernel_size, padding='same',activation='relu',input_shape=(28,28,1))) 

    # Reference for tf.keras.layers.MaxPooling2D : https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D
    # tf.keras.layers.MaxPooling2D : Max pooling operation for 2D spatial data. (reduces input representation dimensionality)
    # 2D convolution window of (2,2)  
    self.model.add(tf.keras.layers.MaxPooling2D(pool_size= (2, 2))) 

    # Use dropout and an L2 penalty for regularization.
    # https://towardsdatascience.com/regularization-in-deep-learning-l1-l2-and-dropout-377e75acc036

    # Reference for tf.keras.layers.Dropout :https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
    # tf.keras.layers.Dropout : sets a fraction rate of input units as 0 for each update to prevent overfitting by downscale factor
    self.model.add(tf.keras.layers.Dropout(0.50))

    # Reference for tf.keras.layers.Flatten : https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten
    # tf.keras.layers.Flatten : Flattens input without affecting batch size. 
    self.model.add(tf.keras.layers.Flatten())  

    # Reference for tf.keras.layers.Dense :  https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
    # tf.keras.layers.Dense : Dense implements the operation: output = activation(dot(input, kernel) + bias) 
    # activation =  element-wise activation function passed as the activation argument
    # kernel = weights matrix created by the layer 
    # bias =  bias vector created by the layer (only applicable if use_bias is True).
    # With tf.keras.regularizers :  https://keras.io/api/layers/regularizers/ 

    #** wants probabilty as output**  
    # activation = softmax/ kera 
    # Softmax as the activation for the last layer of a classification network to interprest the result as a probability distribution.
    # Reference for activation = 'softmax': https://keras.io/api/layers/activations/#:~:text=Softmax%20is%20often%20used%20as,odds%20of%20the%20resulting%20probability.      
    self.model.add(tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001))) 

    # Configures the model for training         
    self.model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']) 
    self.model.summary() 
    
# Activate convolution neural network 
# Running Sequential -> 2D convolution -> 2D max_pooling -> Drop out-> Flatten-> Dense (L2) 
# Over 5 epochs 
# https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python
# https://www.thetopsites.net/article/52355114.shtml
if __name__ == "__main__": 
  image_data = Data()     
  CNNmodel = Model()        
  history = CNNmodel.model.fit(image_data.train_feature, image_data.train_label, batch_size = Batch_size, epochs = Epoch_number, validation_data = (image_data.valid_feature, image_data.valid_label)) 
  test_loss, test_accuracy = CNNmodel.model.evaluate(image_data.test_images, image_data.test_labels, verbose = 0) 
  
# Accuracy and Loss on test set
 test_loss, test_accuracy = CNNmodel.model.evaluate(image_data.test_images, image_data.test_labels, verbose = 0) 
      
print("Test set loss is", test_loss)     
print("Test set Accuracy is", test_accuracy) 
plt.figure(1, figsize=[30,30])     
plt.subplot(121)     
plt.plot(history.history['accuracy'])     
plt.plot(history.history['val_accuracy'])     
plt.title('Model Accuracy for Training and Validation Set')    
plt.ylabel('Accuracy')     
plt.xlabel('Epoch')     
plt.legend(['Training', 'Validation'], loc='upper left') 
plt.subplot(122)     
plt.plot(history.history['loss'])    
plt.plot(history.history['val_loss'])     
plt.title('Model loss for Training and Validation Set')     
plt.ylabel('Loss')     
plt.xlabel('Epoch')     
plt.legend(['Training', 'Validation'], loc='upper left')     
plt.show() 
 
