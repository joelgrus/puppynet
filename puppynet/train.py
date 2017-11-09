"""
We train a neural net by repeatedly using an
optimizer and backpropagating gradients
"""

from puppynet.tensor import Tensor
from puppynet.nn import NeuralNet
from puppynet.data import DataIterator, BatchIterator
from puppynet.optim import Optimizer, SGD
from puppynet.loss import Loss, MSE


def train(net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD()) -> None:
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(epoch, epoch_loss)
