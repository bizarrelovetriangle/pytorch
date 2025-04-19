import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pathlib
from torchinfo import summary

def imshow(img, unnormalize = True, transpose = True):
    if unnormalize:
        img = img / 2 + 0.5     # unnormalize
    npimg = img.detach().cpu().data.numpy()
    if transpose:
        npimg = np.transpose(npimg, (1, 2, 0))
    plt.axis('off')
    plt.style.use('dark_background')
    plt.imshow(npimg)
    plt.show()

latentSize = 100
imageWidth = 178
imageHeight = 218
batch_size = 128
path = "C:\\Users\\PC\\Downloads\\celeba"
netPath = pathlib.Path(__file__).parent.resolve()

class EncoderNet:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 68, 5, stride=3),
            nn.MaxPool2d(2, 2),
            nn.Flatten(1),
            nn.Linear(612, 200),
            nn.ReLU(),
            nn.Linear(200, latentSize),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        return self.model(input)

class DecoderNet:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Unflatten(-1, (latentSize, 1, 1)),
            nn.ConvTranspose2d(latentSize, 72, 5, stride = 1, padding=0, output_padding=0),
            nn.BatchNorm2d(72),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(72, 36, 5, stride = 3, padding=0, output_padding=0),
            nn.BatchNorm2d(36),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(36, 24, 5, stride = 3, padding=(0, 5), output_padding=(0, 0)),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(24, 24, 5, stride = 2, padding=1, output_padding=0),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(24, 3, 5, stride = 2, padding=0, output_padding=1),
            nn.Tanh()
        )
    
    def forward(self, input):
        return self.model(input)

if __name__ == '__main__':
    device = torch.device("cuda:0")

    encoder = EncoderNet()
    encoder.model.to(device)
    encoder.model = nn.DataParallel(encoder.model, list(range(1)))

    decoder = DecoderNet()
    decoder.model.to(device)
    decoder.model = nn.DataParallel(decoder.model, list(range(1)))

    encoder.model.load_state_dict(torch.load(netPath / 'encoder.pth', weights_only=True))
    decoder.model.load_state_dict(torch.load(netPath / 'decoder.pth', weights_only=True))
    print(f'loaded from {netPath}')
    
    image = torchvision.io.read_image("C:\\Users\\PC\\Desktop\\Test.png")
    image = ((image[0:3, :, :].float() / 256) - 0.5) * 2

    images = image[None, :, :, :]
    encoded = encoder.model(images)
    print(encoded.shape)
    output = decoder.model(encoded).detach().cpu()
    print(output.shape)
    cat = torch.cat((images[:1], output[:1]), 0)
    imshow(torchvision.utils.make_grid(cat))

    exit()

    #summary(encoder.model, input_size=(batch_size, 3, imageWidth, imageHeight))
    #summary(decoder.model, input_size=(batch_size, latentSize))
    #exit()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = torchvision.datasets.ImageFolder(root=path, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)


    #dataIter = next(iter(dataloader))
    #batch = dataIter[0]
    #image = batch[0]
    #output = encoder.model(batch)
    #output2 = decoder.model(output)
    #print(image.shape)
    #print(output.shape)
    #print(output2.shape)
    #imshow(torchvision.utils.make_grid(batch, 5))
    #imshow(torchvision.utils.make_grid(output2, 5))
    #exit()

    train = False

    if train == True:
        encoderOpt = torch.optim.Adam(encoder.model.parameters(), betas=(0.5, 0.999))
        decoderOpt = torch.optim.Adam(decoder.model.parameters(), betas=(0.5, 0.999))

        runningLoss = 0.0
        
        for epoch in range(500):
            for i, batch in enumerate(dataloader):
                encoderOpt.zero_grad()
                decoderOpt.zero_grad()

                inputs = batch[0].to(device)
                encoded = encoder.model(inputs)
                output = decoder.model(encoded)
                loss = torch.abs(inputs - output).sum()

                loss.backward()
                encoderOpt.step()
                decoderOpt.step()

                runningLoss += loss.item()
                if i % 100 == 99:
                    print(f'[epoch - {epoch}, {i}] - running loss: {runningLoss / 1000:.3f}')
                    runningLoss = 0.0
            
            if epoch % 10 == 9:
                torch.save(encoder.model.state_dict(), netPath / 'encoder.pth')
                torch.save(decoder.model.state_dict(), netPath / 'decoder.pth')
                print(f'safed to {netPath}')
    else:
        dataIter = next(iter(dataloader))
        batch = dataIter[0]
        image = batch[0]
        output = encoder.model(batch)
        output2 = decoder.model(output).detach().cpu()
        ba = torch.cat((batch[:16], output2[:16]), 0)
        print(ba.shape)
        imshow(torchvision.utils.make_grid(ba, 8))


        #output = decoder.model(torch.rand(24, latentSize))
        #imshow(torchvision.utils.make_grid(output, 6))
