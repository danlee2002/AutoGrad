import numpy as np
from typing import Union, Tuple
class Tensor:
  
  def __init__(self, data: Union[int, float, np.ndarray, list], _children:tuple[type["Tensor"]]=(), _op='', label='') -> type["Tensor"]:
    if isinstance(data, np.ndarray):
      self.data = data
    else:
      self.data = np.array(data)
    self.grad = np.zeros_like(self.data)
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label

  def __add__(self, other: Union[int, float, np.ndarray, list]) -> type["Tensor"]:
    other = self.checktype(other)
    out = Tensor(self.data + other.data, (self, other), '+')
    def _backward():
      self.grad = self.grad + (1.0 * out.grad)
      other.grad = other.grad + (1.0 * out.grad)
    out._backward = _backward
    return out

  def __mul__(self, other: Union[int, float, np.ndarray, list]) -> type["Tensor"]:
    other = self.checktype(other)
    out = Tensor(self.data * other.data, (self, other), _op = '*')
    def _backward():
      self.grad = self.grad + (other.data * out.grad)
      other.grad = other.grad + (self.data * out.grad)
    out._backward = _backward
    return out

  def __pow__(self, other: Union[int, float, np.ndarray, list]) -> type["Tensor"]:
    if isinstance(other, (int, float)):
      out_data = np.power(self.data, other)
      out = Tensor(out_data, (self,), _op = f'**{other}')
      def _backward():
        self.grad = self.grad + ((other * np.power(self.data, other - 1)) * out.grad)

      out._backward = _backward
      return out
    else:
      out_data = np.power(self.data, other.data)
      out = Tensor(out_data, (self, other), _op = f'**')
      def _backward():
        self.grad = self.grad + ((other.data * np.power(self.data * other.data - 1)) * out.grad)
        other.grad = other.grad + (np.log(self.data) * out.grad)
      out._backward = _backward
      return out
 
  def __radd__(self, other: Union[int, float, np.ndarray, list]) -> type["Tensor"]:
    return self + other

  def __rmul__(self, other: Union[int, float, np.ndarray, list]) -> type["Tensor"]:
    return self * other

  def __truediv__(self, other: Union[int, float, np.ndarray, list]) -> type["Tensor"]:
    return self * other**-1

  def __neg__(self) -> type["Tensor"]:
    return self * -1

  def __sub__(self, other: Union[int, float, np.ndarray, list]) -> type["Tensor"]:
    return self + (-other)

  def exp(self) -> type["Tensor"]:
    x = self.data
    out = Tensor(np.exp(x), (self, ), _op = 'exp')
    def _backward():
        self.grad = self.grad + (out.data * out.grad)
    out._backward = _backward
    return out
  def sum(self):
    ...
  def tanh(self) -> type["Tensor"]:
    x = self.data
    t = np.tanh(x)
    out = Tensor(t, (self, ), 'tanh')
    def _backward():
        self.grad = self.grad + ((1 - t**2) * out.grad)
    out._backward = _backward
    return out

  def relu(self) -> type["Tensor"]:
    x = self.data 
    out = Tensor(np.maximum(x, 0), (self,), _op = 'relu')
    def _backward():
      self.grad = self.grad + np.where(x >=0, 1, 0) * out.grad
    out._backward = _backward
    return out 
  
  def sigmoid(self) -> type["Tensor"]:
    x = self.data
    sig = 1/(1+np.exp(-self.data))
    out = Tensor(sig, (self,), _op = 'sigmoid')
    def _backward():
      self.grad = self.grad + sig*(1-sig)*out.grad
    out._backward = _backward
    return out 

  def log(self) -> type["Tensor"]:
    data = np.log(self.data)
    out = Tensor(data, (self,), _op = 'log')
    def _backward():
      self.grad = self.grad + 1/data* out.grad
    out._backward = _backward
    return out

  def backward(self):
    topo = []
    visited = set()
    def toposort(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                toposort(child)
            topo.append(v)
    toposort(self)
    self.grad = np.ones_like(self.data)
    for node in reversed(topo):
        node._backward()

  # util functions 
  def sum(self) -> float:
    return self.data.sum()

  def eye(n: int) -> type['Tensor']:
    return Tensor(np.eye(n))
  
  def __repr__(self):
    return f"Value(data={self.data})"
  
  # checks if input is Tensor and converts otherwise
  def checktype(self, other: Union[int, float, np.ndarray, list]) -> type["Tensor"]:
    return other if isinstance(other, Tensor) else Tensor(other)

