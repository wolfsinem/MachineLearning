import numpy as np 
import random

"""
Artifical Neural Network is an information processing paradigm that is inspired by 
the human brain. Neurons are connected together by synapses. 

In Computer Science we model this exact process by creating networks using matrices. 
We will model a simple NN to solve XOR. 

The XOR (exclusive or) function should return a true value if the two inputs are not equal and a false value
if they are equal. 

input 1 | input 2 | output 
--------------------------
   0    |    0    |  0
   0    |    1    |  1
   1    |    1    |  0
   1    |    0    |  1

   https://github.com/Ricky-N/NeuralNetwork-XOR/blob/master/xor.py

"""

"""Sigmoid functions used to introduce nonlinearity in the model"""
def sigmoid_function(self,x):
    """sigmoid function for hidden layer"""
    return 1.0/(1.0+np.exp(-x))
    
def sigmoid_derivative(self,x):
    """derivative of the sigmoid function"""
    return x * (1 - x)
        
class Neuron:
    """
    representing each neuron in the network.
        - for every input, multiply by its weight
        - sum all of the weighted inputs
        - compute the output w sigmoid 
    """

    def __init__(self, input_neuron, weight_neuron):
        """
        param_input_neuron = given input in the first layer
        param_weight = factor of how important each neuron is
        """

        self.input_neuron = input_neuron 
        self.weight_neuron = weight_neuron

        self.bias = rand()
        self.last_output = 0
        self.last_input = []

    
    def get_output(self,inputs):
        """output of a neuron f = (w1 * X1 +/- bias) + (w2 * X2 +/- bias)"""

        self.last_input = inputs

        output = 0
        for i in range(len(inputs)):
            output = output + inputs[i] * self.weight_neuron[i]
        output = output + self.bias

        self.last_output = sigmoid_function(output)
        return self.last_output

class Neural_network:
    """Initializing the Neural Network"""

    def __init__(self, input_layer = 2, hidden_layer = 2, output_layer= 1):
        """all the layers in the neural network"""
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer

        hidden_weights = np.random.uniform(size=(self.input_layer,self.hidden_layer))