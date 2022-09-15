#Libraries
import tensorflow as tf
import numpy as np
#Network class
class Network(object):
    def __init__(self, layers):
        self.weights = []
        for i in range(len(layers)-1):
            self.weights.append(tf.random.uniform([layers[i+1],layers[i]],minval=-10,maxval=10,
                    dtype=tf.dtypes.float32,seed=None,name=None))
        self.biases = []
        for i in range(len(layers)-1):
            self.biases.append(tf.random.uniform([layers[i+1]],minval=-10,maxval=10,dtype=tf.dtypes.float32,seed=None,name=None))
    #feedforward function, missing activation function rn
    def feedforward(self, a):
        for i in range(len(self.weights)):
            a = self.tanh(tf.add(tf.linalg.matvec(self.weights[i],a),self.biases[i]))
        return(a)
#hyperbolyc tangent
    def tanh(self, z):
        return(1-(2.0/(1+np.exp(-2*z))))
#derivative of hyperbolic tangent
    def tanh_prime(self, z):
        return(1-tanh(z)**2)
#ReLU function
    def ReLU(self, z):
        if z < 0:
            return self.alpha*z
        else:
            return z
#derivative of ReLU
    def ReLU_prime(self, z):
        if z < 0:
            return self.alpha
        else:
            return 1

nn = Network([5,10,10,10,1])
print(nn.feedforward([1.0,1.0,1.0,1.0,1.0]))
