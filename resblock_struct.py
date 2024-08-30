import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
print(model)
model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
print(model)
