from __future__ import annotations
import numpy as np
from typing import Union

class Tensor:

  def __init__(self, data: Union[int, float, np.ndarray, list], _children:tuple[Tensor]=(), _op='', label='',dtype = np.float64, _requiresgrad:bool = True) -> Tensor:
    if isinstance(data, np.ndarray):
      self.data = data
    else:
      self.data = np.array(data)
    self.grad = np.zeros_like(self.data)
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label
    self._requiresgrad = _requiresgrad

  def __add__(self, other: Union[int, float, np.ndarray, list]) -> Tensor:
    other = self.checktype(other)
    out = Tensor(self.data + other.data, (self, other) if self._requiresgrad else (), '+')

    def _backward():
      self.grad = self.grad + (1.0 * out.grad)
      other.grad = other.grad + (1.0 * out.grad)
    if self._requiresgrad: out._backward = _backward
    return out

  def __mul__(self, other: Union[int, float, np.ndarray, list]) -> Tensor:
    other = self.checktype(other)
    out = Tensor(self.data * other.data, (self, other) if self._requiresgrad else (), _op = '*')
    def _backward():
      self.grad = self.grad + (other.data * out.grad)
      other.grad = other.grad + (self.data * out.grad)
    if self._requiresgrad: out._backward = _backward
    return out

  def __pow__(self, other: Union[int, float, np.ndarray, list]) -> Tensor:
    if isinstance(other, (int, float)):
      out_data = np.power(self.data, other)
      out = Tensor(out_data, (self,) if self._requiresgrad else(), _op = f'**{other}')
      def _backward():
        self.grad = self.grad + ((other * np.power(self.data, other - 1)) * out.grad)
      if self._requiresgrad: out._backward = _backward
      return out
    else:
      out_data = np.power(self.data, other.data)
      out = Tensor(out_data, (self, other) if self._requiresgrad else (), _op = f'**')
      def _backward():
        self.grad = self.grad + ((other.data * np.power(self.data * other.data - 1)) * out.grad)
        other.grad = other.grad + (np.log(self.data) * out.grad)
      if self._requiresgrad: out._backward = _backward 
      return out
 
  def __radd__(self, other: Union[int, float, np.ndarray, list]) -> Tensor:
    return self + other

  def __rmul__(self, other: Union[int, float, np.ndarray, list]) -> Tensor:
    return self * other

  def __truediv__(self, other: Union[int, float, np.ndarray, list]) -> type[Tensor]:
    return self * other**-1

  def __neg__(self) -> type[Tensor]:
    return self * -1

  def __sub__(self, other: Union[int, float, np.ndarray, list]) -> type[Tensor]:
    return self + (-other)

  def exp(self) -> type[Tensor]:
    x = self.data
    out = Tensor(np.exp(x), (self, ) if self._requiresgrad else (), _op = 'exp')
    def _backward():
        self.grad = self.grad + (out.data * out.grad)
    if self._requiresgrad: out._backward = _backward
    return out

  def tanh(self) -> Tensor:
    x = self.data
    t = np.tanh(x)
    out = Tensor(t, (self, ) if self._requiresgrad else (), 'tanh')
    def _backward():
        self.grad = self.grad + ((1 - t**2) * out.grad)
    if self._requiresgrad: out._backward = _backward
    return out

  def relu(self) -> Tensor:
    x = self.data 
    out = Tensor(np.maximum(x, 0), (self,) if self._requiresgrad else (), _op = 'relu')
    def _backward():
      self.grad = self.grad + np.where(x >=0, 1, 0) * out.grad
    if self._requiresgrad: out._backward = _backward
    return out 
  
  def sigmoid(self) -> Tensor:
    sig = 1/(1+np.exp(-self.data))
    out = Tensor(sig, (self,) if self._requiresgrad else (), _op = 'sigmoid')
    def _backward():
      self.grad = self.grad + sig*(1-sig)*out.grad
    if self._requiresgrad: out._backward = _backward
    return out 

  def log(self) -> Tensor:
    data = np.log(self.data)
    out = Tensor(data, (self,) if self._requiresgrad else (), _op = 'log')
    def _backward():
      self.grad = self.grad + np.ones_like(data)/self.data* out.grad
    if self._requiresgrad: out._backward = _backward
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

  def eye(n: int) -> Tensor:
    return Tensor(np.eye(n))
  
  def arange(start, stop, step = 1):
    return Tensor(np.arange(start, stop,step))


  def __repr__(self):
    return f"Value(data={self.data})"
  
  # checks if input is Tensor and converts otherwise
  def checktype(self, other: Union[int, float, np.ndarray, list]) -> Tensor:
    return other if isinstance(other, Tensor) else Tensor(other)
  

