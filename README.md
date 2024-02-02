# AutoGrad
Autograd is an automatic differentiation and machine library that aim to provide an api similar to PyTorch  and Andrej Karpathy's minigrad. At the current moment it supports MLP and there are plans to add features such as CNNs and LSTMs. 

# Example Usage Learning halfspace
```python
from autograd.neuralnetwork import nn,MLP
model = nn([MLP(2,4,10,2)], loss = nn.crossentropy, lr = 0.2)


xs = [[-1,-2],[-1,-1],[1,3],[1,1],[2,1],[-3,-1],[-2,-2]]
ys = [[1,0],[1,0],[0,1],[0,1],[1,0],[1,0]]
for i in range(400):
  y_pred = model.forward(xs)
  model.backward(y_pred, ys)
```

More sample usage can be found[here](tests)
