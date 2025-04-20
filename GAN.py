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

class SkipConnection(torch.nn.Module):
    def __init__(self, features):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(features),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(features, features, 5, stride = 1, padding=(2, 2), output_padding=(0, 0)),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(features, features, 5, stride = 1, padding=(2, 2), output_padding=(0, 0)),
        )
    
    def forward(self, input):
        return self.model(input) + input

class DiscriminatorNet:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 24, 5, stride=2),
            nn.ReLU(),
            SkipConnection(24),
            nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2),
            SkipConnection(36),
            nn.ReLU(),
            nn.Conv2d(36, 68, 5, stride=3),
            SkipConnection(68),
            nn.Conv2d(68, 1, kernel_size=(7, 5)),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        return self.model(input)

class EncoderNet:
    def __init__(self):
        self.model = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 24, 5, stride=2),
            nn.ReLU(),
            SkipConnection(24),
            nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2),
            SkipConnection(36),
            nn.ReLU(),
            nn.Conv2d(36, 68, 5, stride=3),
            SkipConnection(68),
            nn.Conv2d(68, latentSize, kernel_size=(7, 5)),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        return self.model(input)

class DecoderNet:
    def __init__(self):
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latentSize, 72, 5, stride = 1, padding=0, output_padding=0),
            nn.BatchNorm2d(72),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(72, 36, 5, stride = 3, padding=0, output_padding=0),
            SkipConnection(36),
            nn.BatchNorm2d(36),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(36, 24, 5, stride = 3, padding=(0, 5), output_padding=(0, 0)),
            SkipConnection(24),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(24, 24, 5, stride = 2, padding=1, output_padding=0),
            SkipConnection(24),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(24, 3, 5, stride = 2, padding=0, output_padding=1),
            nn.Tanh()
        )
    
    def forward(self, input):
        return self.model(input)

if __name__ == '__main__':
    device = torch.device("cuda:0")

    discriminator = DiscriminatorNet()
    discriminator.model.to(device)
    discriminator.model = nn.DataParallel(discriminator.model, list(range(1)))

    encoder = EncoderNet()
    encoder.model.to(device)
    encoder.model = nn.DataParallel(encoder.model, list(range(1)))

    decoder = DecoderNet()
    decoder.model.to(device)
    decoder.model = nn.DataParallel(decoder.model, list(range(1)))

    #summary(discriminator.model, input_size=(batch_size, 3, imageHeight, imageWidth))
    #summary(encoder.model, input_size=(batch_size, 3, imageHeight, imageWidth))
    #summary(decoder.model, input_size=(batch_size, latentSize, 1, 1))
    #exit()

    #discriminator.model.load_state_dict(torch.load(netPath / 'discriminator.pth', weights_only=True))
    #encoder.model.load_state_dict(torch.load(netPath / 'encoder.pth', weights_only=True))
    #decoder.model.load_state_dict(torch.load(netPath / 'decoder.pth', weights_only=True))
    #print(f'loaded from {netPath}')
    

    #image = torchvision.io.read_image("C:\\Users\\PC\\Desktop\\Test.png")
    #image = ((image[0:3, :, :].float() / 256) - 0.5) * 2
    #images = image[None, :, :, :]
    #encoded = encoder.model(images)
    #print(encoded.shape)
    #output = decoder.model(encoded).detach().cpu()
    #print(output.shape)
    #cat = torch.cat((images[:1], output[:1]), 0)
    #imshow(torchvision.utils.make_grid(cat))
    #exit()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = torchvision.datasets.ImageFolder(root=path, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    train = True

    if train == True:
        discriminatorCriterion = nn.BCELoss()
        discriminatorOpt = torch.optim.Adam(discriminator.model.parameters(), betas=(0.5, 0.999))
        encoderOpt = torch.optim.Adam(encoder.model.parameters(), betas=(0.5, 0.999))
        decoderOpt = torch.optim.Adam(decoder.model.parameters(), betas=(0.5, 0.999))

        cycleConsistentLoss = 0.0
        discriminatorLoss = 0.0
        
        for epoch in range(1):
            for i, batch in enumerate(dataloader):
                discriminatorOpt.zero_grad()
                encoderOpt.zero_grad()
                decoderOpt.zero_grad()

                inputs = batch[0].to(device)

                encoded = encoder.model(inputs)
                output = decoder.model(encoded)

                # labels
                trueLabels = torch.full((inputs.size(0), 1, 1, 1), 1, dtype=torch.float32, device=device)
                fakeLabels = torch.full((inputs.size(0), 1, 1, 1), 0, dtype=torch.float32, device=device)

                # calculating losses for the discriminator
                discriminatorReals = discriminator.model(inputs)
                discriminatorRealError = discriminatorCriterion(discriminatorReals, trueLabels)
                discriminatorRealError.backward()
                
                discriminatorFakeOutputs = discriminator.model(output.detach())
                discriminatorFakeError = discriminatorCriterion(discriminatorFakeOutputs, fakeLabels)
                discriminatorFakeError.backward()

                #discriminatorOpt.step()

                # calculating losses for the generator
                #cycleConsistentError = torch.abs(inputs - output).sum()
                #cycleConsistentError.backward(retain_graph=True)

                discriminatorFakeOutputs2 = discriminator.model(output)
                discriminatorFakeError2 = discriminatorCriterion(discriminatorFakeOutputs2, trueLabels)
                discriminatorFakeError2.backward()

                encoderOpt.step()
                decoderOpt.step()

                discriminatorLoss += discriminatorRealError.item() + discriminatorFakeError.item()
                #cycleConsistentLoss += cycleConsistentError.item()
                #if i % 100 == 99:
                print(f'[epoch - {epoch}, {i}/{len(dataloader)}] - cycleConsistentLoss: {cycleConsistentLoss / 100:>10.3f}, discriminatorLoss: {discriminatorLoss / 100:<10.3f}')
                print(f'Discriminator real mean: {discriminatorReals.view(-1).mean():.6f}, Discriminator fake mean: {discriminatorFakeOutputs.view(-1).mean():.6f}')
                print(f'discriminatorRealError: {discriminatorRealError:.6f}, discriminatorFakeError: {discriminatorFakeError:.6f}')
                discriminatorLoss = 0.0
                cycleConsistentLoss = 0.0
            
    #if epoch % 10 == 9:
        torch.save(discriminator.model.state_dict(), netPath / 'discriminator.pth')
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
