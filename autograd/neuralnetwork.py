from engine import Tensor
from typing import List, Union
import random
import numpy as np
from collections import OrderedDict
class Neuron:

  def __init__(self, nin: int, _activation = Tensor.tanh):
    self.w = [Tensor(random.uniform(-1, 1)) for _ in range(nin)]
    self.b = Tensor(random.uniform(-1,1))
    self._activation = _activation
  def __call__(self, x):
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    out = act.tanh()
    return out
  
  def parameters(self):
    return self.w + [self.b]
  
class Layer:

  def __init__(self, nin: int, nout: int, _activation =  Tensor.tanh):
    self.neurons = [Neuron(nin, _activation = _activation) for _ in range(nout)]
  
  def __call__(self, x) -> Union[List[type[Tensor]],type[Tensor]]:
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs
  
  def parameters(self):
    # Return the parameters of all neurons in the layer
    return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:

  def __init__(self, nin,hsize, layers, nouts, _activation = Tensor.tanh):
    self.layers = [Layer(nin, hsize, _activation)] + [Layer(hsize, hsize, _activation) for layer in range(layers)] + [Layer(hsize, nouts,_activation)]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]



"""
container class for neural networks
"""
class nn:
  def __init__(self, containers, loss, lr = 1e-3, trackLoss = True):
    self.modules = containers 
    self.lr = lr
    self.loss = loss
    self.trackLoss = trackLoss
    self.losslist = []

  def forward(self, x):
    y = []
    for x_elem in x:
      y_pred = x_elem
      for module in self.modules:
        y_pred = module(y_pred)
      y = y + [y_pred]
    return y

  def backward(self, y_pred, y_true):
    loss = self.loss(y_pred,y_true)
    for module in self.modules:
      for p in module.parameters():
        p.grad = 0
      loss.backward()
      for p in module.parameters():
        p.data += -self.lr * p.grad

  def mse(y_pred,y_true):
   return sum((yout - ygt) ** 2 for ygt, yout in zip(y_true, y_pred))



