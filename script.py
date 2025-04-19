import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights




input1 = torch.ones(1, 5, 5) #torch.Tensor(([[1, 1], [1, 1]],[[1, 1], [1, 1]],[[1, 1], [1, 1]]))
#print(input1.shape)
model1 = torch.nn.Conv2d(1, 1, 2, stride=1, padding=1, bias=False)
nn.init.ones_(model1.weight)
output1 = model1(input1)
print(input1)
print(output1)

#input2 = torch.ones(100)
#model2 = torch.nn.ConvTranspose2d( 100, 64 * 8, 4, 1, 0, bias=False)
#output2 = model2(input2)


output = torch.ones(2, 2)



def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

"""""
model = resnet18(weights=ResNet18_Weights.DEFAULT)
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

loss = (model(data) - labels).sum()
loss.backward()
optim.step()

#print(res)
"""""

"""""
a = torch.tensor([1., 2.], requires_grad=True)
b = torch.tensor([1., 2.], requires_grad=True)
print(a*b)
Q = a**2 * b

gradient = torch.tensor([1., 2.])
Q.backward(gradient=gradient)
print(a.grad == 2*a*b * gradient)
#imshow(torchvision.utils.make_grid(data))
"""""