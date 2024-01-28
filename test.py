import torch 
from autograd.engine import Tensor

a = torch.tensor([2.0]); a.requires_grad = True
b = torch.sigmoid(a)
b.backward()
print(a.grad.item())

a1 = Tensor(2.0)
b1 = Tensor.sigmoid(a1)
b1.backward()
print(a1.grad)
