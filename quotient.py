import torch

n = 23
incc= 32
x = torch.tensor(n, requires_grad=True, dtype=torch.float)
y = x/(incc+x)
y.backward()
print(x.grad)
print(f"{1/(incc+n)-n/(incc+n)**2:.4f}")
