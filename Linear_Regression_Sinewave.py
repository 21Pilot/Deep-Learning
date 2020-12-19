# -*- coding: utf-8 -*-
# Flora Seo 
# 2020/9/9
# Deep Learning_Project1

# Set Up 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
N , M = 50 , 6 

# Data
x_train = np.random.sample( N ) * 8 - 4  #  -4 < x < 4
n_train = np.random.normal(0, 0.1, size=N)  #  noise (μ,σ) = (0,0.1)
sin_train = np.sin( x_train ) + n_train  # tf.sin(x)
print( x_train.shape , n_train.shape , sin_train.shape ) 

# Plot
plt.figure( figsize = ( 10 , 5 ) )
plt.subplot( 1 , 2 , 1 )
plt.scatter( x_train , sin_train , color = 'red', marker = "o")  
plt.title( "tensorflow.sin" )  
plt.xlabel( "X" )  
plt.ylabel( "Y" )
x = np.linspace( -4 , 4.0 , N )
plt.plot( x , np.sin( x ) )
plt.subplot( 1 , 2 , 2 )  #  1-column 1-row plot's 2nd plot
m , std = np.linspace( -3 , 3.0 , M ) , np.array( [ 1.0 ] * M )
for i in range( M ) :
    plt.plot( x , np.exp( -( ( x - m[ i ] ) / std[ i ] ) ** 2 ) )
plt.show( )

# Gaussian
gaussian_m = m.reshape( -1 , 1 )
gaussian_std = std.reshape( -1 , 1 )
print( m.shape , std.shape , gaussian_m.shape , gaussian_std.shape )
g_w = np.array( [ 0. , -1. , 0. , 0. , 1. , 0. ] ).reshape( -1 , 1 )
gaussian_w = tf.Variable( initial_value = g_w , dtype = tf.float64 , trainable = True )  #  initial weight
gaussian_b = tf.Variable( tf.zeros( [ M , 1 ] , dtype = tf.float64 ) , trainable = True )
print( gaussian_w , gaussian_b , x_train.shape , gaussian_m.shape )
gaussian_exp = tf.exp( - tf.square( ( x_train - gaussian_m ) / gaussian_std ) )
print( gaussian_w.shape , gaussian_b.shape , gaussian_exp.shape )
yhat = tf.reduce_sum( gaussian_exp * gaussian_w + gaussian_b , 0 )
print( gaussian_exp.shape , yhat.shape )
loss = tf.reduce_sum( tf.square( sin_train - yhat ) )
print( yhat.shape , loss.shape )

# Gradient Descnet Optimizer
optimizer = tf.compat.v1.train.GradientDescentOptimizer( 0.5 )
tf.disable_v2_behavior()
train = optimizer.minimize( loss )
init = tf.initialize_all_variables( )  #  init variable
ss = tf.Session( )  #  launch graph
ss.run( tf.variables_initializer( optimizer.variables( ) ) )
ss.run( init )
for step in range( 20 ) :
    ss.run( train )
    print( step , ss.run( gaussian_w ) , ss.run( gaussian_b ) , ss.run
( loss ) )
ss.close( )
