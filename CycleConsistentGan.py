import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pathlib
import struct
from torchinfo import summary
import os
import datetime
from pathlib import Path
from common import SchedulerParams
from common import Common

# Root directory for dataset
dataroot = "data/celeba"

chanels = 3

gFeaturesCount = 64
dFeaturesCount = 64

latentSize = 100

imageWidth = 80
imageHeight = 80

batch_size = 128

alignDataSetPath = Path("/home/rrasulov/training_data/align_dataset")
unalignDataSetPath = Path("/home/rrasulov/training_data/unalign_dataset")
#modelsPath = pathlib.Path(__file__).parent.resolve()
runDataPath = Path("/home/rrasulov/run_data")
modelsPath = "models"
intermediatePath = "intermediete_images"
modelLoadPath = Path("/home/rrasulov/run_data/20250518-173528/models")

class SkipConnection(torch.nn.Module):
    def __init__(self, features, slope = 0.01):
        super().__init__()
        self.model = nn.Sequential(
            nn.LazyConv2d(features, 3, stride=1, padding=1),
            nn.LeakyReLU(slope),
            nn.LazyConv2d(features, 3, stride=1, padding=1),
            nn.LazyBatchNorm2d(),
        )
    
    def forward(self, input):
        return self.model(input) + input

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.LazyConv2d(dFeaturesCount, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),

            nn.LazyConv2d(dFeaturesCount * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.2, inplace=False),

            nn.LazyConv2d(dFeaturesCount * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.2, inplace=False),

            nn.LazyConv2d(dFeaturesCount * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.2, inplace=False),
            
            nn.LazyConv2d(dFeaturesCount * 8, kernel_size=4, stride=2, padding=1, bias=False),
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
            nn.LazyConv2d(gFeaturesCount, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.02, inplace=False),

            SkipConnection(gFeaturesCount, 0.02),
            nn.LeakyReLU(0.02),

            nn.LazyConv2d(gFeaturesCount * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.02, inplace=False),

            SkipConnection(gFeaturesCount * 2, 0.02),
            nn.LeakyReLU(0.02),

            nn.LazyConv2d(gFeaturesCount * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.02, inplace=False),

            SkipConnection(gFeaturesCount * 4, 0.02),
            nn.LeakyReLU(0.02),

            nn.LazyConv2d(gFeaturesCount * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.02, inplace=False),
            
            #SkipConnection(gFeaturesCount * 8, 0.02),
            #nn.LeakyReLU(0.02),
            
            nn.LazyConv2d(latentSize, kernel_size=5, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        return self.main(input)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.LazyConvTranspose2d(gFeaturesCount * 8, kernel_size=5, stride=1, padding=0, bias=False),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.02, inplace=False),
            
            #SkipConnection(gFeaturesCount * 8, 0.02),
            #nn.LeakyReLU(0.02),

            nn.LazyConvTranspose2d(gFeaturesCount * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.02, inplace=False),

            SkipConnection(gFeaturesCount * 4, 0.02),
            nn.LeakyReLU(0.02),

            nn.LazyConvTranspose2d(gFeaturesCount * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.02, inplace=False),

            SkipConnection(gFeaturesCount * 2, 0.02),
            nn.LeakyReLU(0.02),

            nn.LazyConvTranspose2d(gFeaturesCount, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.02, inplace=False),

            SkipConnection(gFeaturesCount, 0.02),
            nn.LeakyReLU(0.02),

            nn.LazyConvTranspose2d(chanels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, input):
        return self.main(input)
    
def lr_scheduler(loss, ideal_loss, x_min, x_max, h_min=0.1, f_max=2.0):
  x = np.abs(loss-ideal_loss)
  f_x = np.clip(np.pow(f_max, x/x_max), 1.0, f_max)
  h_x = np.clip(np.pow(h_min, x/x_min), h_min, 1.0)
  return f_x if loss > ideal_loss else h_x

def learn(discriminator, encoder, generator, dLosses, gLosses):
    #discriminator.load_state_dict(torch.load(modelLoadPath / 'discriminator.pth', weights_only=True))
    #generator.load_state_dict(torch.load(modelLoadPath / 'generator.pth', weights_only=True))
    #dLosses = load_integer_list(modelLoadPath / "dLosses.list")
    #gLosses = load_integer_list(modelLoadPath / "gLosses.list")
    #print(f'loaded from {modelLoadPath}')

    discriminator.to(device)
    discriminator.apply(Common.weights_init)

    encoder.to(device)
    encoder.apply(Common.weights_init)

    generator.to(device)
    generator.apply(Common.weights_init)
    
    runName = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    currentRunPath = runDataPath / runName
    currentModelsPath = currentRunPath / modelsPath 
    intermediateDataPath = currentRunPath / intermediatePath
    currentModelsPath.mkdir(parents=True, exist_ok=True)
    intermediateDataPath.mkdir(parents=True, exist_ok=True)

    transform=transforms.Compose([
        transforms.Resize((imageHeight, imageWidth)),
        transforms.CenterCrop((imageHeight, imageWidth)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = torchvision.datasets.ImageFolder(root=alignDataSetPath, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    discriminatorCriterion = nn.BCELoss()

    lr = 0.0001 #😫
    beta1 = 0.5 #momentumCoef
    beta2 = 0.999 #decayRate
    ### m = beta1*m + (1-beta1)*dx
    ### cache = beta2*cache + (1-beta2)*(dx**2)
    ### x += - learning_rate * m / (np.sqrt(cache) + eps)
    discriminatorOpt = torch.optim.Adam(discriminator.parameters(), lr=lr * 0.2, betas=(beta1, beta2))
    generatorOpt = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    encoderOpt = torch.optim.Adam(encoder.parameters(), lr = lr, betas=(0.5, 0.999))

    cycleConsistentLoss = 0.0
    discriminatorLoss = 0.0

    for epoch in range(300):
        for i, data in enumerate(dataloader):
            inputs = data[0].to(device)
            
            encoder.zero_grad()
            generator.zero_grad()

            encoded = encoder(inputs)
            generated = generator(encoded)

            cycleConsistentError = torch.abs(torch.square(inputs - generated)).sum()
            cycleConsistentError.backward()

            encoderOpt.step()
            generatorOpt.step()

            cycleConsistentLoss += cycleConsistentError.item()
            
            if i % 100 == 99:
                print(f'[epoch - {epoch}, {i}/{len(dataloader)}] - cycleConsistentLoss: {cycleConsistentLoss / 100:>10.3f}')
                cycleConsistentLoss = 0.0

                with torch.no_grad():
                    intermediateImagePath = intermediateDataPath / f'{epoch}_{i}.png' 
                    unnormalizedData = torch.cat((inputs[:8].cpu(), generated[:8].detach().cpu()), 0) / 2 +0.5
                    intermediateData = torchvision.utils.make_grid(unnormalizedData, 4)
                    torchvision.utils.save_image(intermediateData, intermediateImagePath)
                    print(f'epoch {epoch} is over, intermediate images are stored as \'{intermediateImagePath}\'')

            # b_size = inputs.size(0)
            # #fakeLabels = torch.pow(torch.rand((b_size,), dtype=torch.float32, device=device, requires_grad=False) / 2, 3)
            # #realLabels = 1 - torch.pow(torch.rand((b_size,), dtype=torch.float32, device=device, requires_grad=False) / 2, 3)
            # realLabels = torch.full((b_size,), 1, dtype=torch.float32, device=device, requires_grad=False)
            # fakeLabels = torch.full((b_size,), 0, dtype=torch.float32, device=device, requires_grad=False)

            # noise = torch.randn(b_size, latentSize, 1, 1, device=device)
            # generated = generator(noise)

            # discriminatorOpt.zero_grad()
            # discriminatorRealsOutput = discriminator(inputs).view(-1)
            # discriminatorRealError = discriminatorCriterion(discriminatorRealsOutput, realLabels)
            # discriminatorFakeOutputs = discriminator(generated.detach()).view(-1)
            # discriminatorFakeError = discriminatorCriterion(discriminatorFakeOutputs, fakeLabels)

            # discriminatorError = (discriminatorRealError + discriminatorFakeError)
            # discriminatorError.backward()
            # discriminatorOpt.step()

            # generatorOpt.zero_grad()
            # discriminatorGeneratorOutputs = discriminator(generated).view(-1)
            # discriminatorGeneratorError = discriminatorCriterion(discriminatorGeneratorOutputs, realLabels)
            # discriminatorGeneratorError.backward()
            # generatorOpt.step()


            # dLosses.append(discriminatorError.item())
            # gLosses.append(discriminatorGeneratorError.item())
            # if i % 100 == 99:
            #     print(f'[epoch - {epoch}, {i}/{len(dataloader)}]')
            #     #print(f'cycleConsistentLoss: {cycleConsistentLoss / 100:>10.3f}, discriminatorLoss: {discriminatorLoss / 100:<10.3f}')
            #     print(f'discriminator real (m/e): {discriminatorRealsOutput.mean():.6f} / {discriminatorRealError:.6f}')
            #     print(f'discriminator fake (m/e): {discriminatorFakeOutputs.mean():.6f} / {discriminatorFakeError:.6f}')
            #     print(f'discriminatorGenerator (m/e): {discriminatorGeneratorOutputs.mean():.6f} / {discriminatorGeneratorError:.6f}')
            #     #discriminatorLoss = 0.0
        
            #     with torch.no_grad():
            #         intermediateImagePath = intermediateDataPath / f'{epoch}_{i}.png' 
            #         intermediateImages = generator(torch.randn(8, latentSize, 1, 1, device=device))
            #         unnormalizedData =  intermediateImages.detach().cpu() / 2 + 0.5
            #         intermediateData = torchvision.utils.make_grid(unnormalizedData, 4)
            #         torchvision.utils.save_image(intermediateData, intermediateImagePath)
            #         print(f'epoch {epoch} is over, intermediate images are stored as \'{intermediateImagePath}\'')
        
        if epoch % 5 == 0:
            Common.save_integer_list(dLosses, currentModelsPath / "dLosses.list")
            Common.save_integer_list(gLosses, currentModelsPath / "gLosses.list")
            print(f'losses safed to {currentModelsPath}')
            torch.save(discriminator.state_dict(), currentModelsPath / 'discriminator.pth')
            torch.save(generator.state_dict(), currentModelsPath / 'generator.pth')
            torch.save(encoder.state_dict(), currentModelsPath / 'encoder.pth')
            print(f'models safed to {currentModelsPath}')

if __name__ == '__main__':
    runDataPath.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")

    discriminator = Discriminator()
    encoder = Encoder()
    generator = Generator()

    gLosses = []
    dLosses = []

    summary(discriminator, input_size=(batch_size, 3, imageHeight, imageWidth))
    summary(encoder, input_size=(batch_size, 3, imageHeight, imageWidth))
    summary(generator, input_size=(batch_size, latentSize, 1, 1))
    exit()

    train = 0
    
    if train == 0:
        learn(discriminator, encoder, generator, dLosses, gLosses)
    elif train == 1:
        Common.legend(dLosses, gLosses)
    else:
        Common.showupGan(Path("/home/rrasulov/nntest"), modelLoadPath, generator, latentSize)
        Common.showupCycleMyData(Path("/home/rrasulov/nntest"), Path("/home/rrasulov/nntest"), modelLoadPath, encoder, generator)
