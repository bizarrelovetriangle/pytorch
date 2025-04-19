import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pathlib
from torchinfo import summary

def imshow(img, unnormalize = True):
    if unnormalize:
        img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().cpu().data.numpy()
    plt.axis('off')
    plt.style.use('dark_background')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class Net:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Conv2d(3, 36, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(36, 24, 5),
            nn.MaxPool2d(2, 2),
            nn.Flatten(1),
            nn.Linear(24 * 5 * 5, 200),
            nn.ReLU(),
            nn.Linear(200, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
    
    def forward(self, input):
        return self.model(input)

if __name__ == '__main__':
    batch_size = 5
    path = pathlib.Path(__file__).parent.resolve() / "classifier.pth"

    classifier = Net()
    
    classifier.model.load_state_dict(torch.load(path, weights_only=True))
    summary(classifier.model, input_size=(batch_size, 3, 32, 32))
    exit()
    #weightsTensor = classifier.model[0].weight
    #print(weightsTensor.shape)
    #windows = weightsTensor.split(1, 0)
    #windows = list({windows[i].flatten(0, 1) for i, w in enumerate(windows)})
    #print(windows[0].shape)
    #imshow(torchvision.utils.make_grid(windows, 6, 1))
    #exit()


    #exit()

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    train = False

    if train == True:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(classifier.model.parameters(), lr=0.0008, momentum=0.8)
        #optimizer = torch.optim.Adam(classifier.model.parameters(), betas=(0.5, 0.999))

        runningLoss = 0.0

        for epoch in range(5):
            for i, butchData in enumerate(trainloader, 0):
                inputs, labels = butchData
                optimizer.zero_grad()
                outputs = classifier.model(inputs)

                labelsTensor = torch.nn.functional.one_hot(labels, 10)
                diff = (outputs - labelsTensor)
                loss = (diff * diff).sum()
                #loss = torch.abs(torch.flatten((outputs - labelsTensor))).sum()
                #loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                runningLoss += loss.item()
                if i % 100 == 0:
                    print(f'[epoch - {epoch}, {i}] - running loss: {runningLoss / 1000:.3f}')
                    runningLoss = 0.0

        torch.save(classifier.model.state_dict(), path)
        print(f'safed to {path}')
    else:
        classifier.model.load_state_dict(torch.load(path, weights_only=True))
        print(f'loaded from {path}')

        classesCount = {i:0 for i in range(10)}
        classesMatch = {i:0 for i in range(10)}

        for i, butchData in enumerate(testloader, 0):
            inputs, labels = butchData
            outputs = classifier.model(inputs)
            predicted = outputs.argmax(1)
            for i, l in enumerate(labels, 0):
                labelInt = l.item()
                classesCount[labelInt] += 1
                if predicted[i] == labelInt:
                    classesMatch[labelInt] += 1
        print(f'total match - {sum(classesMatch.values()):>7}/{sum(classesCount.values()):<7} - {sum(classesMatch.values()) / sum(classesCount.values()) * 100:.2f}%')
        print(f'\n\r'.join(f'{classes[i]:8s} - {classesMatch[i]:>5}/{classesCount[i]:<5} - {classesMatch[i] / classesCount[i] * 100:>6.2f}%' for i in classesCount.keys()))

        #output = classifier.model(images).argmax(1)
        #print('labels: ' + ' '.join(f'{classes[j]:10s}' for j in labels))
        #print('output: ' + ' '.join(f'{classes[j]:10s}' for j in output))
        #imshow(torchvision.utils.make_grid(images))
        

        
"""""
    output = classifier.model(images)
    print(output)
    # show images
    print(anImage.shape)
    flatenModel = nn.Flatten(0, 1)
    print(flatenModel(anImage).shape)
    test = nn.Conv2d(3, 6, 5)
    print(test(images[0]).shape)
    test = nn.Sequential(nn.Conv2d(3, 6, 5), nn.MaxPool2d(2, 2))
    print(test(images[0]).shape)
    #imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:10s}' for j in range(batch_size)))
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