# AutoGrad
Autograd is an automatic differentiation library written with the purpose of simplifying the calculation of deriavtives.
It aims an api similar to PyTorch and utilizes the Reverse-Mode variant of automatic differentiation.
At the current moment, it has built in support for FCNN and there are plans to add more advanced architechtures in the future such as CNN and LSTM.
### Example usage
```python
from autograd.engine import Tensor
# initalizes a Tensor size of (n,) 
a = Tensor(data = [2,2,2])
b = Tensor(data = [1,1,1])
# performs elementwise multiplication
c = a * b 
# calcualtes partial in respect to 
c.backward()
# prints partial of a in respect 
print(a.grad)
```