import math

from layer import Layer
from network import Network
from math import tanh


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

network = Network(2)

layer = Layer(10, 0, tanh)
network.addLayer(layer)
layer = Layer(1, 0, sigmoid)
network.addLayer(layer)

output = network.getOutput([0.0,0.1])
print(output)