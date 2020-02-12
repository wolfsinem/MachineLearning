from typing import TYPE_CHECKING, List
from connection import Connection


class Neuron:
    def __init__(self, bias: float, activation):
        self.connections : List[Connection] = []
        self.prev_layer = None
        self.output : float = 0
        self.bias : float = bias
        self.activation = activation
        self.input = False

    def fire(self):
        total = 0
        if self.input:
            for connection in self.connections:
                total += connection.value * connection.weight
        else:
            for connection in self.connections:
                total += connection.neuron_from.output * connection.weight
        output = total * self.bias
        self.output = self.activation(output)

    def connect_layer(self, layer):
        self.prev_layer = layer
        for neuron in layer.neurons:
            connection = Connection(neuron, 0)
            self.connections.append(connection)
    
    def create_input(self, input_size):
        for i in range(input_size):
            connection = Connection(None, 0)
            self.connections.append(connection)
            self.input = True
    
    def set_input(self, f_input: List[float]):
        for index, data in enumerate(f_input):
            self.connections[index].value = data

    def set_weights(self, weights: List[float]):
        for index, weight in weights:
            self.connections[index].weight = weight