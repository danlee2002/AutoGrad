# AutoGrad
Autograd is an automatic differentiation library written with the purpose of simplifying the calculation of deriavtives.
It aims an api similar to PyTorch and utilizes the Reverse-Mode variant of automatic differentiation.
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