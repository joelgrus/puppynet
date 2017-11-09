"""
Can't learn xor with a linear model
"""

import numpy as np

from puppynet.nn import NeuralNet
from puppynet.layers import Linear, Tanh
from puppynet.train import train

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

net = NeuralNet([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2)
])

train(net, inputs, targets)

for x, y in zip(inputs, targets):
    print(x, net.forward(x), y)
