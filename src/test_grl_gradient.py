import torch
from grl import grad_reverse

x = torch.tensor([[1.0, 2.0, 3.0]],requires_grad=True)
y = grad_reverse(x,strength=1.0)
loss = y.sum()
loss.backward()
print("Gradient:",x.grad)