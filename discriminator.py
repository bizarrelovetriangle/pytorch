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
dFeaturesCount = 24

latentSize = 100

imageWidth = 128
imageHeight = 128

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
            # nn.LazyConv2d(dFeaturesCount, 6, stride=3, padding=2, bias=False), # 43
            # nn.LeakyReLU(0.02, inplace=False),
    
            # nn.LazyConv2d(dFeaturesCount * 2, kernel_size=6, stride=3, padding=1, bias=False), # 14
            # nn.LazyBatchNorm2d(),
            # nn.LeakyReLU(0.02, inplace=False),
    
            # nn.LazyConv2d(dFeaturesCount * 4, kernel_size=6, stride=2, padding=(0, 0), bias=False), # 5
            # nn.LazyBatchNorm2d(),
            # nn.LeakyReLU(0.02, inplace=False),

            # nn.LazyConv2d(1, kernel_size=5, stride=1, padding=0, bias=False),
            # nn.Sigmoid()
            
            nn.LazyConv2d(dFeaturesCount, kernel_size=6, stride=2, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=False),

            nn.LazyConv2d(dFeaturesCount * 2, kernel_size=6, stride=2, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=False),

            nn.LazyConv2d(dFeaturesCount * 4, kernel_size=5, stride=2, padding=0, bias=False),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.2, inplace=False),

            nn.LazyConv2d(dFeaturesCount * 8, kernel_size=5, stride=2, padding=0, bias=False),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.2, inplace=False),

            nn.LazyConv2d(1, kernel_size=5, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        return self.main(input)
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.LazyConv2d(gFeaturesCount, kernel_size=6, stride=2, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=False),

            nn.LazyConv2d(gFeaturesCount * 2, kernel_size=6, stride=2, padding=0, bias=False),
            nn.LeakyReLU(0.2, inplace=False),

            nn.LazyConv2d(gFeaturesCount * 4, kernel_size=5, stride=2, padding=0, bias=False),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.2, inplace=False),

            nn.LazyConv2d(gFeaturesCount * 8, kernel_size=5, stride=2, padding=0, bias=False),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.02, inplace=False),

            nn.LazyConv2d(latentSize, kernel_size=5, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        return self.main(input)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.LazyConvTranspose2d(gFeaturesCount * 8, kernel_size=(5, 5), stride=1, padding=0, bias=False),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.02, inplace=False),

            nn.LazyConvTranspose2d(gFeaturesCount * 4, kernel_size=(5, 5), stride=2, padding=0, bias=False),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.02, inplace=False),

            SkipConnection(gFeaturesCount * 4),
            nn.LeakyReLU(0.02, inplace=False),

            SkipConnection(gFeaturesCount * 4),
            nn.LeakyReLU(0.02, inplace=False),

            nn.LazyConvTranspose2d(gFeaturesCount * 2, kernel_size=(5, 5), stride=2, padding=0, bias=False),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.02, inplace=False),

            SkipConnection(gFeaturesCount * 2),
            nn.LeakyReLU(0.02, inplace=False),

            SkipConnection(gFeaturesCount * 2),
            nn.LeakyReLU(0.02, inplace=False),

            nn.LazyConvTranspose2d(gFeaturesCount, kernel_size=6, stride=2, padding=0, bias=False),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.02, inplace=False),

            SkipConnection(gFeaturesCount),
            nn.LeakyReLU(0.02, inplace=False),

            SkipConnection(gFeaturesCount),
            nn.LeakyReLU(0.02, inplace=False),

            SkipConnection(gFeaturesCount),
            nn.LeakyReLU(0.02, inplace=False),

            SkipConnection(gFeaturesCount),
            nn.LeakyReLU(0.02, inplace=False),

            nn.LazyConvTranspose2d(chanels, kernel_size=6, stride=2, padding=0, bias=False),
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

def learn(discriminator, encoder, generator, dLosses, gLosses):
    #discriminator.load_state_dict(torch.load(netPath / 'discriminator.pth', weights_only=True))
    #generator.load_state_dict(torch.load(netPath / 'generator.pth', weights_only=True))
    #dLosses = load_integer_list(netPath / "dLosses.list")
    #gLosses = load_integer_list(netPath / "gLosses.list")
    #print(f'loaded from {netPath}')

    transform=transforms.Compose([
        transforms.Resize((imageHeight, imageWidth)),
        transforms.CenterCrop((imageHeight, imageWidth)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = torchvision.datasets.ImageFolder(root=path, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    discriminatorCriterion = nn.BCELoss()

    lr = 0.0001 #😫
    beta1 = 0.5 #momentumCoef
    beta2 = 0.999 #decayRate
    ### m = beta1*m + (1-beta1)*dx
    ### cache = beta2*cache + (1-beta2)*(dx**2)
    ### x += - learning_rate * m / (np.sqrt(cache) + eps)
    discriminatorOpt = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    generatorOpt = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))

    cycleConsistentLoss = 0.0
    discriminatorLoss = 0.0
    smoothed_disc_loss = scheduler_params.ideal_loss

    for epoch in range(300):
        for i, data in enumerate(dataloader):
            inputs = data[0].to(device)
            b_size = inputs.size(0)
            
            #fakeLabels = torch.pow(torch.rand((b_size,), dtype=torch.float32, device=device, requires_grad=False) / 2, 3)
            #realLabels = 1 - torch.pow(torch.rand((b_size,), dtype=torch.float32, device=device, requires_grad=False) / 2, 3)
            realLabels = torch.full((b_size,), 1, dtype=torch.float32, device=device, requires_grad=False)
            fakeLabels = torch.full((b_size,), 0, dtype=torch.float32, device=device, requires_grad=False)

            noise = torch.randn(b_size, latentSize, 1, 1, device=device)
            generated = generator(noise)


            discriminatorOpt.zero_grad()
            discriminatorRealsOutput = discriminator(inputs).view(-1)
            discriminatorRealError = discriminatorCriterion(discriminatorRealsOutput, realLabels)
            discriminatorFakeOutputs = discriminator(generated.detach()).view(-1)
            discriminatorFakeError = discriminatorCriterion(discriminatorFakeOutputs, fakeLabels)

            discriminatorError = (discriminatorRealError + discriminatorFakeError) / 2
            discriminatorError.backward()
            discriminatorOpt.step()

            generatorOpt.zero_grad()
            discriminatorGeneratorOutputs = discriminator(generated).view(-1)
            discriminatorGeneratorError = discriminatorCriterion(discriminatorGeneratorOutputs, realLabels)
            discriminatorGeneratorError.backward()
            generatorOpt.step()


            dLosses.append(discriminatorError.item())
            gLosses.append(discriminatorGeneratorError.item())
            if i % 100 == 99:
                print(f'[epoch - {epoch}, {i}/{len(dataloader)}]')
                #print(f'cycleConsistentLoss: {cycleConsistentLoss / 100:>10.3f}, discriminatorLoss: {discriminatorLoss / 100:<10.3f}')
                print(f'discriminator re al (m/e): {discriminatorRealsOutput.mean():.6f} / {discriminatorRealError:.6f}')
                print(f'discriminator fake (m/e): {discriminatorFakeOutputs.mean():.6f} / {discriminatorFakeError:.6f}')
                print(f'discriminatorGenerator (m/e): {discriminatorGeneratorOutputs.mean():.6f} / {discriminatorGeneratorError:.6f}')
                #print(f'generator lr: {lr}, discriminator lr: {dynamicLR}')
                #discriminatorLoss = 0.0
        
    save_integer_list(dLosses, netPath / "dLosses.list")
    save_integer_list(gLosses, netPath / "gLosses.list")
    print(f'losses safed to {netPath}')

    torch.save(discriminator.state_dict(), netPath / 'discriminator.pth')
    torch.save(generator.state_dict(), netPath / 'generator.pth')
    print(f'models safed to {netPath}')


def showup(generator):
    generator.load_state_dict(torch.load(netPath / 'generator.pth', weights_only=True))
    print(f'loaded from {netPath}')

    output2 = generator(torch.randn(16, latentSize, 1, 1, device=device))
    imshow(torchvision.utils.make_grid(output2.detach().cpu(), 8))

if __name__ == '__main__':
    device = torch.device("cuda:0")

    discriminator = Discriminator()
    discriminator.to(device)
    discriminator.apply(weights_init)

    encoder = Encoder()
    encoder.to(device)
    encoder.apply(weights_init)

    generator = Generator()
    generator.to(device)
    generator.apply(weights_init)
    
    gLosses = []
    dLosses = []

    summary(discriminator, input_size=(batch_size, 3, imageHeight, imageWidth))
    summary(generator, input_size=(batch_size, latentSize, 1, 1))
    exit()

    train = 0
    
    if train == 0:
        learn(discriminator, encoder, generator, dLosses, gLosses)
    elif train == 1:
        legend(dLosses, gLosses)
    else:
        showup(generator)
