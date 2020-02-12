from typing import TYPE_CHECKING, List, Optional
from neuron import Neuron


if TYPE_CHECKING:
    from connection import Connection


class Layer:
    def __init__(self, neuron_amount : int, bias: int, activation):
        self.neurons : List[Neuron] = []
        self.activation = activation
        self.bias = bias
        self.create_neurons(neuron_amount)
        self.input = False

    @property
    def output(self):
        total = 0
        for neuron in self.neurons:
            total += neuron.output
        return total

    def create_neurons(self, neuron_amount):
        for i in range(0, neuron_amount):
            neuron = Neuron(self.bias, self.activation)
            self.neurons.append(neuron)

    def connect_layer(self, prev_layer):
        for neuron in self.neurons:
            neuron.connect_layer(prev_layer)
    
    def fire(self):
        for neuron in self.neurons:
            neuron.fire()

    def set_as_input(self, input_size: float):
        for neuron in self.neurons:
            neuron.create_input(input_size)
    
    def set_input(self, f_input : List[float]):
        for neuron in self.neurons:
            neuron.set_input(f_input)
        self.input = True

    def set_weights(self, weights : List[List[float]]):
        for index, neuron in enumerate(self.neurons):
            neuron.set_weights(weights[index])