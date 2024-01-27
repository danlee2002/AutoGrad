import numpy as np 
import math
from typing import Union, Tuple

class Tensor:
    """
    Initializes Tensor class from np.ndarray object or list
    """
    def __init__(self, data: Union[int, float, np.ndarray, list], _children: Tuple[type["Tensor"]] = (), _op = '', label:str = ''):
        if isinstance(data, np.ndarray, ):
            self.data = data
        else: 
            self.data = np.array(data)
        #initalizes gradient
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        #sets nodes of parents 
        self.prev = set(_children)
        #operation associated with current nodes
        self._op = _op
        #label for visualization 
        self.label = label 

    def __repr__(self):
        return f'Value: {self.data}'

    def __truediv__(self, other: Union[int, float, np.ndarray, type["Tensor"]]) -> type["Tensor"]:
        ...

    def __add__(self, other:Union[int, float, np.ndarray, type["Tensor"]]) -> type["Tensor"]:
        other = self.checktype(other)
        out = Tensor(data = self.data + other.data, _children = (self,other), _op = '+')
        def _backward():
            self.grad = self.grad + (1.0 * out.grad)
            other.grad = other.grad + (1.0 * out.grad)
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other

    def __mul__(self, other: Union[int, float, np.ndarray, type["Tensor"]]) ->  type["Tensor"]:
        other = self.checktype(other)
        out = Tensor(data = self.data * other.data, _children = (self, other), _op = '*')
        def _backward():
            self.grad = self.grad + (other.data * out.grad)
            other.grad = other.grad + (self.data * out.grad)
        out._backward = _backward
        return out 
   
   
    def __matmul__(self, other: Union[int, float, np.ndarray, type["Tensor"]]) -> type["Tensor"]:
        other =  self.checktype(other)
        ...
    

    def __pow__(self, other: Union[int, float, np.ndarray, type["Tensor"]])  -> type["Tensor"]:
        # handles scalar case 
        if isinstance(other, (int, float)):
            out = Tensor(data =self.data**other, _children= (self,), _op = '**')
            def _backward():
                self.grad = self.grad + (other * np.power(self.data, other - 1) * out.grad)
            out._backward = _backward
            return out 
        else:
            other = self.checktype(other)
            out = Tensor(data = np.power(self.data,other.data),_children= (self, other), _op = '**')
            def _backward():
                self.grad = self.grad + (other.data * np.power(self.data, other.data - 1) * out.grad)
                other.grad= other.grad + (np.log(self.data) * out.grad)
            out._backward = _backward
            return out 

    def __neg__(self) -> type["Tensor"]:
        return self * -1

    def __sub__(self, other: Union[float, int, np.ndarray,type["Tensor"]]) ->type["Tensor"] :
        return self + (-other)

    def __truediv__(self, other: Union[float, int, np.ndarray,type["Tensor"]]) -> type["Tensor"]:
        return self * other ** -1

    def relu(self):
        out = Tensor(data = np.maximum(self.data,0), _children = (self,), _op = "relu")
        def _backward():
            self.grad = self.grad + np.where(self.data >= 0, 1, 0)*out.grad
        self._backward = _backward
        return out

    def tanh(self):
        value = np.tanh(self.data)
        out = Tensor(data = value, _children = (self,), _op = "tanh")
        def _backward():
            self.grad = self.grad + (1 - value**2) * out.grad
        self._backward = _backward
        return out 
    
    def sigmoid(self):
        sgmd = 1/(1 + np.exp(self.data))
        out = Tensor(data = sgmd, _children = (self,),_op = "sigmoid")
        def _backward():
            self.grad = self.grad + (1 - sgmd)* sgmd
        self._backward = _backward
        return out 
    
        

    def checktype(self, other: any):
        return other if isinstance(other,Tensor) else Tensor(other) 
    




    def backward(self):
        topo = []
        visited = set()
        def topoSort(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    topoSort(child)
            topo.append(v)
       
        topoSort(self)
        self.grad = np.ones_like(self.data)
        for nodes in reversed(topo):
            nodes._backward()

