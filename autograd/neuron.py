import random
from engine import Tensor
from typing import Union, Tuple
class Neuron:
    def __init__(self, nin, activation = Tensor.relu):
        self.w = [Tensor(random.uniform(-1,1)) for _ in range(nin)]
        self.activation = activation 

