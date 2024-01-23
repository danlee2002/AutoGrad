import numpy as np 
import math
import matplotlib.pyplot as plt

class Tensor:

    def __init__(self, data, child = (), op = "", label =''):
        self.data = data
        self.grad = 0 
        self.child= set(child)
        self.op = op
        self.label = label

    def __repr__(self):
        return f'value: {self.data}'
    
    def __add__(self, other):
        return Tensor(self.data + other.data, (self, other), "+")
    
    def __mul__(self, other):
        return Tensor(self.data * other.data, (self, other), "*")
    
    def __truediv__(self, other):
        return Tensor(self.data / other.data,  (self, other), "/")
    



