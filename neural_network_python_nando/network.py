from typing import TYPE_CHECKING, List
from layer import Layer


class Network:
    def __init__(self, input_size: int):
        self.layers: List[Layer] = []
        self.input_size = input_size
        pass

    def fire(self, f_input : List[float]):
        self.layers[0].set_input(f_input)
        for layer in self.layers:
            layer.fire()
        self.output = self.layers[-1].output

    def getOutput(self, f_input : List[float]):
        self.fire(f_input)
        return self.output
    
    def addLayer(self, layer : Layer):
        self.layers.append(layer)
        if len(self.layers) > 1:
            layer.connect_layer(self.layers[-2])
        else:
            layer.set_as_input(self.input_size)
