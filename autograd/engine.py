import numpy as np 
import math
from typing import Union, Tuple

class Tensor:
    """
    Initializes Tensor class from np.ndarray object or list
    """
    def __init__(self, data: Union[np.ndarray, list], _children: Tuple[type["Tensor"]] = (), _op = '', label:str = ''):
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

    def __truediv__(self, other: type["Tensor"]) -> type["Tensor"]:
        ...

    def __add__(self, other: type["Tensor"]) -> type["Tensor"]:
        other = self.checktype(other)
        out = Tensor(data = self.data + other.data, _children = (self,other), _op = '+')
        def _backward():
            self.grad = self.grad + (1.0 * out.grad)
            other.grad = other.grad + (1.0 * out.grad)
        out._backward = _backward
        return out

    def __mul__(self, other: type["Tensor"]) ->  type["Tensor"]:
        other = self.checktype(other)
        out = Tensor(data = self.data * other.data, _children = (self, other), _op = '*')
        def _backward():
            self.grad = self.grad + (other.data * out.grad)
            other.grad = other.grad + (self.data * out.grad)
        out._backward = _backward
        return out 
   
   
    def __matmul__(self, other):
        other =  self.checktype(other)
        ...
    

    def __pow__(self, other: Union[int, float, type["Tensor"]])  -> type["Tensor"]:
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



y = Tensor([2,2,2], label ="b")
d =Tensor([3,3,3], label ="c") 
c = y**d
c.backward()

print(y.grad)