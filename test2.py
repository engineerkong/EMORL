import torch
a = torch.randn(4, 4)
r = torch.argmax(a, dim=-1)
print(a, r)