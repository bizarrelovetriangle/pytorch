import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pathlib
import struct
from torchinfo import summary

def save_integer_list(int_list, file_path):
    with open(file_path, 'wb') as f:
        f.write(struct.pack('I', len(int_list)))
        f.write(struct.pack(f'{len(int_list)}f', *int_list))

def load_integer_list(file_path):
    with open(file_path, 'rb') as f:
        list_length = struct.unpack('I', f.read(4))[0]
        int_list = list(struct.unpack(f'{list_length}f', f.read(4 * list_length)))
    return int_list

def legend(dLosses, gLosses):
    dLosses = load_integer_list(netPath / "dLosses.list")
    gLosses = load_integer_list(netPath / "gLosses.list")

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(gLosses,label="G")
    plt.plot(dLosses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Root directory for dataset
dataroot = "data/celeba"

chanels = 3

gFeaturesCount = 32
dFeaturesCount = 8

latentSize = 100

imageWidth = 111
imageHeight = 111

batch_size = 128

path = "C:\\Users\\PC\\Downloads\\celeba"
netPath = pathlib.Path(__file__).parent.resolve()


class SkipConnection(torch.nn.Module):
    def __init__(self, features):
        super().__init__()
        self.model = nn.Sequential(
            nn.LazyConvTranspose2d(features, 5, stride = 1, padding=(2, 2)),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(),
            nn.LazyConvTranspose2d(features, 5, stride = 1, padding=(2, 2)),
            nn.LazyBatchNorm2d(),
        )
    
    def forward(self, input):
        return self.model(input) + input

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
        )
    def forward(self, input):
        return self.main(input)
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.LazyConv2d(gFeaturesCount, kernel_size=6, stride=3, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=False),

            SkipConnection(gFeaturesCount),
            nn.LeakyReLU(0.02, inplace=False),

            nn.LazyConv2d(gFeaturesCount * 2, kernel_size=6, stride=3, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=False),

            SkipConnection(gFeaturesCount * 2),
            nn.LeakyReLU(0.02, inplace=False),

            SkipConnection(gFeaturesCount * 2),
            nn.LeakyReLU(0.02, inplace=False),

            nn.LazyConv2d(gFeaturesCount * 4, kernel_size=5, stride=2, padding=0, bias=False),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.02, inplace=False),

            SkipConnection(gFeaturesCount * 4),
            nn.LeakyReLU(0.02, inplace=False),
            
            SkipConnection(gFeaturesCount * 4),
            nn.LeakyReLU(0.02, inplace=False),
            
            SkipConnection(gFeaturesCount * 4),
            nn.LeakyReLU(0.02, inplace=False),

            nn.LazyConv2d(latentSize, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Tanh()
        )
        
    def forward(self, input):
        return self.main(input)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.LazyConvTranspose2d(gFeaturesCount * 4, kernel_size=4, stride=1, padding=0, bias=False), # kernel_size=5 ??
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.02, inplace=False),

            SkipConnection(gFeaturesCount * 4),
            nn.LeakyReLU(0.02, inplace=False),
            
            SkipConnection(gFeaturesCount * 4),
            nn.LeakyReLU(0.02, inplace=False),

            nn.LazyConvTranspose2d(gFeaturesCount * 2, kernel_size=5, stride=2, padding=0, bias=False),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.02, inplace=False),

            SkipConnection(gFeaturesCount * 2),
            nn.LeakyReLU(0.02, inplace=False),

            nn.LazyConvTranspose2d(gFeaturesCount, kernel_size=6, stride=3, padding=0, bias=False),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.02, inplace=False),

            SkipConnection(gFeaturesCount),
            nn.LeakyReLU(0.02, inplace=False),

            nn.LazyConvTranspose2d(chanels, kernel_size=6, stride=3, padding=0, bias=False),
            nn.Tanh()
        )
        
    def forward(self, input):
        return self.main(input)
    
class SchedulerParams:
    ideal_loss = np.log(4)
    x_min = 0.1 * np.log(4)
    x_max = 0.1 * np.log(4)
    h_min = 0.1
    f_max = 2.0
    
scheduler_params = SchedulerParams()

def lr_scheduler(loss, ideal_loss, x_min, x_max, h_min=0.1, f_max=2.0):
  x = np.abs(loss-ideal_loss)
  f_x = np.clip(np.pow(f_max, x/x_max), 1.0, f_max)
  h_x = np.clip(np.pow(h_min, x/x_min), h_min, 1.0)
  return f_x if loss > ideal_loss else h_x

def learn(discriminator, encoder, decoder, dLosses, gLosses):
    #discriminator.load_state_dict(torch.load(netPath / 'discriminator.pth', weights_only=True))

    encoder.load_state_dict(torch.load(netPath / 'encoder.pth', weights_only=True))
    decoder.load_state_dict(torch.load(netPath / 'generator.pth', weights_only=True))
    dLosses = load_integer_list(netPath / "dLosses.list")
    gLosses = load_integer_list(netPath / "gLosses.list")
    print(f'loaded from {netPath}')

    transform=transforms.Compose([
        transforms.Resize((imageHeight, imageWidth)),
        transforms.CenterCrop((imageHeight, imageWidth)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = torchvision.datasets.ImageFolder(root=path, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    lr = 0.0002
    #discriminatorCriterion = nn.BCELoss()
    #discriminatorOpt = torch.optim.Adam(discriminator.parameters(), lr = lr, betas=(0.5, 0.999))
    encoderOpt = torch.optim.Adam(encoder.parameters(), lr = lr, betas=(0.5, 0.999))
    decoderOpt = torch.optim.Adam(decoder.parameters(), lr = lr, betas=(0.5, 0.999))

    cycleConsistentLoss = 0.0
    
    for epoch in range(200):
        for i, batch in enumerate(dataloader):
            inputs = batch[0].to(device)

            encoder.zero_grad()
            decoder.zero_grad()

            encoded = encoder(inputs)
            generated = decoder(encoded)

            cycleConsistentError = torch.abs(inputs - generated).sum()
            cycleConsistentError.backward()

            encoderOpt.step()
            decoderOpt.step()

            cycleConsistentLoss += cycleConsistentError.item()
            
            if i % 100 == 99:
                print(f'[epoch - {epoch}, {i}/{len(dataloader)}] - cycleConsistentLoss: {cycleConsistentLoss / 100:>10.3f}')
                cycleConsistentLoss = 0.0
                    
    save_integer_list(dLosses, netPath / "dLosses.list")
    save_integer_list(gLosses, netPath / "gLosses.list")
    print(f'losses safed to {netPath}')

    #torch.save(discriminator.state_dict(), netPath / 'discriminator.pth')
    torch.save(encoder.state_dict(), netPath / 'encoder.pth')
    torch.save(decoder.state_dict(), netPath / 'generator.pth')
    print(f'models safed to {netPath}')


def showup():
    decoder = Generator()
    decoder.load_state_dict(torch.load(netPath / 'generator.pth', weights_only=True))
    print(f'loaded from {netPath}')

    output2 = decoder(torch.randn(16, latentSize, 1, 1))
    imshow(torchvision.utils.make_grid(output2.detach().cpu(), 8))

def showup2():
    encoder = Encoder()
    decoder = Generator()
    encoder.load_state_dict(torch.load(netPath / 'encoder.pth', weights_only=True))
    decoder.load_state_dict(torch.load(netPath / 'generator.pth', weights_only=True))

    transform=transforms.Compose([
        transforms.Resize((imageHeight, imageWidth)),
        transforms.CenterCrop((imageHeight, imageWidth)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = torchvision.datasets.ImageFolder(root=path, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    dataIter = next(iter(dataloader))
    inputs = dataIter[0]

    encoded = encoder(inputs)
    output = decoder(encoded)

    ba = torch.cat((inputs[:16].cpu(), output[:16].detach().cpu()), 0)
    print(ba.shape)
    imshow(torchvision.utils.make_grid(ba, 8))


if __name__ == '__main__':
    device = torch.device("cuda:0")

    discriminator = Discriminator()
    discriminator.to(device)
    #discriminator.apply(weights_init)

    encoder = Encoder()
    encoder.to(device)
    #encoder.apply(weights_init)

    decoder = Generator()
    decoder.to(device)
    #decoder.apply(weights_init)
    
    gLosses = []
    dLosses = []

    #summary(encoder, input_size=(batch_size, 3, imageHeight, imageWidth))
    #summary(decoder, input_size=(batch_size, latentSize, 1, 1))
    #exit()

    train = 2
    
    if train == 0:
        learn(discriminator, encoder, decoder, dLosses, gLosses)
    elif train == 1:
        legend(dLosses, gLosses)
    elif train == 2:
        showup()
    else:
        showup2()
