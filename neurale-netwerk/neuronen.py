import numpy as np 

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

"""

class Neuron:
    """
    representing each neuron in the network.
    - for every input, multiply by its weight
    - sum all of the weighted inputs
    - compute the output w sigmoid 
    """

    def __init__(self, input_neuron: float, weight: float):
        """
        param_input = given input in the first layer
        param_weight = factor of how important each neuron is
        """

        self.input_neuron = input_neuron 
        self.weight = weight_neuron

    
    def get_output(self,inputs):
        """
        output of a neuron 
        f = (w1 * X1) + (w2 * X2) 

        sum of each column = h 

        """

        for i in range(len(inputs)):





class Neural_network:
    """         """

    def __init__(self, layer_1, layer_2):
        """all the layers in the neural network"""
        self.layer_1 = layer_1
        self.layer_2 = layer_2


    def sigmoid_function(self,x):
        """sigmoid function for hidden layer"""
        return 1.0/(1.0+np.exp(-x))
    
    def sigmoid_derivative(self,x):
        """derivative of the sigmoid function"""
        return x * (1 - x)
