from __future__ import annotations 
from autograd.engine import Tensor
from typing import List, Union
import numpy as np

class Neuron:
  def __init__(self, nin: int, _activation = Tensor.tanh):
    self.w = [Tensor(np.random.uniform(-1,1)) for _ in range(nin)]
    self.b = Tensor(np.random.uniform(-1,1))
    self._activation = _activation

  def __call__(self, x):
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    out = self._activation(act)
    return out
  
  def parameters(self):
    return self.w + [self.b]
  
class Layer:
  def __init__(self, nin: int, nout: int, _activation =  Tensor.tanh):
    self.neurons = [Neuron(nin, _activation = _activation) for _ in range(nout)]
  
  def __call__(self, x) -> Union[List[Tensor],Tensor]:
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs
  
  def parameters(self):
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
To-do:
implementation convolution neural networks
"""
class Convolution2d:
  def __init__(self, nin):
    ...

  def maxpool():
    ...


class CNN:
  def __init__(self, stride = (1,1)):
    self.stride = stride 
  def cross_correlation():
    ...

"""
container class for neural networks
"""
class nn:
  def __init__(self, containers, loss, lr = 1e-3, trackLoss = True, momentum:float = 0.9):
    self.modules = containers 
    self.lr = lr
    self.mode = 'batch'
    self.loss = loss
    self.trackLoss = trackLoss
    self.losslist = []
    self.momentum = momentum

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
    print(loss)
    for module in self.modules:
      for p in module.parameters():
        p.grad = 0
      loss.backward()
      for p in module.parameters():
        p.data -= self.lr * p.grad
        
  def sgd():
    ...

  def crossentropy(y_pred, y_true):
      expvalue = [[y_i.exp() for y_i in y] for y in y_pred]
      denom = [sum(y) for y in expvalue]
      softmax = [[y_i/denomval for y_i in y] for y, denomval in zip(expvalue,denom)]
      acc = 0
      for val, y in zip(np.array([[y_i.data for y_i in y_pred] for y_pred in softmax]),y_true):
        if np.argmax(val) == np.argmax(y):
          acc+=1.0
      acc = acc/len(y_pred)
      print(acc)
      crossentropy = -sum([sum([ y_i * (s_i.log()) for s_i,y_i in zip(s,y)]) for s,y in zip(softmax, y_true)])/len(y_pred)    
      return crossentropy

  def mse(y_pred,y_true):
   return sum((yout - ygt) ** 2 for ygt, yout in zip(y_true, y_pred))/len(y_pred)
  

