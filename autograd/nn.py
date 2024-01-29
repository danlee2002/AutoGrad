from engine import Tensor
from typing import List, Union
import random

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

  def __init__(self, nin: int, hidden: int,layers: int, nout: int, _activation= Tensor.tanh):
    self.layers =  [Layer(nin, hidden)] + [Layer(hidden, hidden) for i in range(layers)] + [Layer(hidden, nout)]

  def __call__(self, x) -> type[Tensor]:
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self) -> List[Tensor]:
    return [p for layer in self.layers for p in layer.parameters()]
  
  