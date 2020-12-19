# -*- coding: utf-8 -*-
# Flora Seo 
# 2020/9/17
# Deep Learning_Project2

# Set Up 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# data set up 
Learning_rate = 0.1
Sample_number = 200
Batch_size = 50
Batch_number =10000

# class to generate spiral graph
# reference : https://www.telesens.co/2017/09/29/spiral_cntk/
# reference : https://glowingpython.blogspot.com/2017/04/solving-two-spirals-problem-with-keras.html
# Generate spiral graph replica 
class Spiral(object):
    def __init__(self):
        np.random.seed(20000)
        sigma = 0.1
        
        # Spiral 1 using sin/cos
        self.dx1 = np.random.uniform(np.pi/4, (np.pi)*4 ,Sample_number)
        self.dy1 = np.array([np.cos(self.dx1)*self.dx1 + np.random.normal(0,sigma, Sample_number),
                            np.sin(self.dx1)*self.dx1 + np.random.normal(0,sigma, Sample_number)]).T
        # Spiral 2 using Spiral 1
        self.dx2 = np.random.uniform(np.pi/4, (np.pi)*4 ,Sample_number)
        self.dy2 = np.array([np.cos(self.dx2 + np.pi)*self.dx2 + np.random.normal(0,sigma, Sample_number),
                            np.sin(self.dx2 + np.pi)*self.dx2 + np.random.normal(0,sigma, Sample_number)]).T
        self.index = np.arange(2*Sample_number)
        
        self.coordinates = np.concatenate((self.dy1, self.dy2))         
        self.labels = np.concatenate((np.zeros(Sample_number), np.ones(Sample_number)))
        
    # Set up Batch selection
    def batch_setup(self):
        batch = np.random.choice(self.index, size = Batch_size)
        return self.coordinates[batch,:], self.labels[batch].flatten()

#Binary classification on the spirals dataset using multilayer perceptron 
# reference : https://www.kaggle.com/androbomb/simple-nn-with-python-multi-layer-perceptron

# Above provided implementing sigmoid for output interpretation 
class Layers(tf.Module):     
    def __init__(self, dimensions_x = 2, dimensions_layer_1 = 30, dimensions_layer_2 = 30): 
        # biases/ weight 
        self.biases =  { 'p1': tf.Variable(tf.random.normal(shape=[dimensions_layer_1]),  dtype= tf.float32),             
                         'p2': tf.Variable(tf.random.normal(shape=[dimensions_layer_2]),  dtype= tf.float32), 
                         'output_p': tf.Variable(tf.zeros(shape=[]), dtype=  tf.float32) 
                        }
        self.weights = { 't1': tf.Variable(tf.random.normal(shape=[dimensions_x, dimensions_layer_1]), dtype= tf.float32),             
                         't2': tf.Variable(tf.random.normal(shape=[dimensions_layer_1, dimensions_layer_2]), dtype=  tf.float32), 
                         'output_t': tf.Variable(tf.random.normal(shape=[dimensions_layer_2, 1]), dtype=  tf.float32) 
                        } 
         
# reference : https://www.oreilly.com/content/building-deep-learning-neural-networks-using-tensorflow-layers/
# reference :https://www.tensorflow.org/api_docs/python/tf/nn/relu6
# Above provided insight to tn.nn.relu/ tf.reduce.mean
    def layer_setup(self, x_coords):
        layer_1 = tf.nn.relu6(tf.add( tf.matmul(x_coords, self.weights['t1']), self.biases['p1']))         
        layer_2 = tf.nn.relu6(tf.add( tf.matmul(layer_1, self.weights['t2']), self.biases['p2'] ))         
        output = tf.add( tf.matmul(layer_2, self.weights['output_t']), self.biases['output_p'])         
        return tf.squeeze(output)
# reference : https://www.ensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
# two independent loss function 
# one was not enough for accurate result 
    def loss(self, coords, labs):
        loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = labs, logits = self.layer_setup(coords))) 
        loss_2 = (.001) * tf.reduce_sum( [tf.nn.l2_loss(var) for var in model.weights.variables] ) 
        loss = loss_1 + loss_2 
        return loss

# implementation of stochastic gradient descent 
# implementation of momentum optimizer 
if __name__ == "__main__": 
        data = Spiral()
        model = Layers()
        #SGD
        optimizer = tf.optimizers.SGD(learning_rate=Learning_rate)
        
        # GradientTape= Record operations for automatic differentiation. 
        with tf.GradientTape() as tape:  
            coords, labs = data.batch_setup() 
            loss_fromBatch = model.loss(coords.astype(np.float32), labs.astype(np.float32))             
            grads = tape.gradient(loss_fromBatch, model.variables)             
            optimizer.apply_gradients(zip(grads, model.variables))
            
        #graph set up
        xAxis = np.linspace(-15,15,150)
        yAxis = np.linspace(-15,15,150) 
        X, Y = np.meshgrid(xAxis, yAxis) 
        Z = np.zeros((150,150))

 for i in range(150):         
            for j in range(150):
                Zarr = np.array([ X[i,j], Y[i,j] ]).reshape(1,2) 
                Zval = model.layer_setup(Zarr.astype(np.float32))             
                Z[i,j] = tf.nn.sigmoid(Zval) 
# reference : https://jakevdp.github.io/PythonDataScienceHandbook/04.04-density-and-contour-plots.html
# plot contour graph 
plt.figure() 
plt.plot(data.dy1[:,0], data.dy1[:,1], 'ob', markeredgecolor="blue")     
plt.plot(data.dy2[:,0], data.dy2[:,1], 'or', markeredgecolor="blue") 
contour_line = plt.contour(X, Y, Z, levels = [0.5], colors = "green")     
plt.clabel(contour_line, inline = 1,font = 5 , colors='green')
plt.legend(["t=0", "t=1"]) 
plt.contourf(X, Y, Z) 
 
plt.title("Multi layer perception on spiral dataset")     
plt.xlabel('x')    
plt.ylabel('y',rotation=0)     
plt.show()
