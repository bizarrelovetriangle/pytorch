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

gFeaturesCount = 32
dFeaturesCount = 32

latentSize = 124

imageSize = 128

batch_size = 128

alignDataSetPath = Path("/home/rrasulov/training_data/align_dataset")
unalignDataSetPath = Path("/home/rrasulov/training_data/unalign_dataset")
#modelsPath = pathlib.Path(__file__).parent.resolve()
runDataPath = Path("/home/rrasulov/run_data")
modelsPath = "models"
intermediatePath = "intermediete_images"
modelLoadPath = Path("/home/rrasulov/run_data/20250815-122215/models")

class Interpolate(torch.nn.Module):
	def __init__(self, size, mode):
			super().__init__()
			self.size = size
			self.mode = mode

	def forward(self, input):
		return nn.functional.interpolate(input, size=self.size, mode=self.mode)

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		
		self.main = nn.Sequential(
			nn.LazyConv2d(gFeaturesCount, kernel_size=3, stride=1, padding=1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			nn.LazyConv2d(gFeaturesCount, kernel_size=4, stride=2, padding=1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),

			nn.LazyConv2d(gFeaturesCount * 2, kernel_size=4, stride=2, padding=1, bias=False),
			nn.LazyBatchNorm2d(),
			nn.LeakyReLU(0.2, inplace=True),

			nn.LazyConv2d(gFeaturesCount * 4, kernel_size=4, stride=2, padding=1, bias=False),
			nn.LazyBatchNorm2d(),
			nn.LeakyReLU(0.2, inplace=True),

			nn.LazyConv2d(gFeaturesCount * 8, kernel_size=4, stride=2, padding=1, bias=False),
			nn.LazyBatchNorm2d(),
			nn.LeakyReLU(0.2, inplace=True),
 
			nn.LazyConv2d(gFeaturesCount * 16, kernel_size=4, stride=2, padding=1, bias=False),
			nn.LazyBatchNorm2d(),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.final = nn.Sequential(
			nn.Flatten(),
			nn.Linear(gFeaturesCount * 16 * 4 * 4, 1, bias=True),
		)

	def forward(self, input):
		B = input.size(0)
		x = self.main(input)
		x = self.final(x)
		return x
	
class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.main = nn.Sequential(
			nn.LazyConv2d(gFeaturesCount, kernel_size=4, stride=2, padding=1, bias=False),
			nn.LazyBatchNorm2d(),
			nn.LeakyReLU(0.02, inplace=True),

			nn.LazyConv2d(gFeaturesCount * 2, kernel_size=4, stride=2, padding=1, bias=False),
			nn.LazyBatchNorm2d(),
			nn.LeakyReLU(0.02, inplace=True),

			nn.LazyConv2d(gFeaturesCount * 4, kernel_size=4, stride=2, padding=1, bias=False),
			nn.LazyBatchNorm2d(),
			nn.LeakyReLU(0.02, inplace=True),

			nn.LazyConv2d(gFeaturesCount * 8, kernel_size=4, stride=2, padding=1, bias=False),
			nn.LazyBatchNorm2d(),
			nn.LeakyReLU(0.02, inplace=True),
 
			nn.LazyConv2d(gFeaturesCount * 16, kernel_size=4, stride=2, padding=1, bias=False),
			nn.LazyBatchNorm2d(),
			nn.LeakyReLU(0.02, inplace=True),
		)

		self.final = nn.Sequential(
			nn.Flatten(),
			nn.Linear(gFeaturesCount * 16 * 4 * 4, latentSize),
			nn.Tanh()
		)
		
	def forward(self, input):
		B = input.size(0)
		x = self.main(input)
		x = self.final(x)
		return x
	
class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.main = nn.Sequential(
			nn.LazyConvTranspose2d(gFeaturesCount * 8, kernel_size=4, stride=2, padding=1, bias=False),
			nn.LazyBatchNorm2d(),
			nn.LeakyReLU(0.02, inplace=True),

			nn.LazyConvTranspose2d(gFeaturesCount * 4, kernel_size=4, stride=2, padding=1, bias=False),
			nn.LazyBatchNorm2d(),
			nn.LeakyReLU(0.02, inplace=True),

			Interpolate(size=(32, 32), mode='nearest'),
			nn.LazyConv2d(gFeaturesCount * 2, kernel_size=3, stride=1, padding=1, bias=False),
			nn.LazyBatchNorm2d(),
			nn.LeakyReLU(0.02, inplace=True),

			Interpolate(size=(64, 64), mode='nearest'),
			nn.LazyConv2d(gFeaturesCount, kernel_size=3, stride=1, padding=1, bias=False),
			nn.LazyBatchNorm2d(),
			nn.LeakyReLU(0.02, inplace=True),
			
			Interpolate(size=(128, 128), mode='nearest'),
			nn.LazyConv2d(gFeaturesCount, kernel_size=3, stride=1, padding=1, bias=False),
			nn.LazyBatchNorm2d(),
			nn.LeakyReLU(0.02, inplace=True),

			nn.LazyConvTranspose2d(chanels, kernel_size=3, stride=1, padding=1, bias=True),
			nn.Tanh()
		)

		self.project = nn.Linear(latentSize, gFeaturesCount * 16 * 4 * 4, bias=True)

	def forward(self, input):
		B = input.size(0)
		x = self.project(input).view(B, gFeaturesCount * 16, 4, 4)
		return self.main(x)
	
def learn(discriminator, encoder, generator, dLosses, gLosses):
	discriminator.to(device)
	discriminator.apply(Common.weights_init)

	encoder.to(device)
	encoder.apply(Common.weights_init)

	generator.to(device)
	generator.apply(Common.weights_init)
	
	#discriminator.load_state_dict(torch.load(modelLoadPath / 'discriminator.pth', weights_only=True), strict=False)
	#generator.load_state_dict(torch.load(modelLoadPath / 'generator.pth', weights_only=True), strict=False)
	#dLosses = Common.load_integer_list(modelLoadPath / "dLosses.list")
	#gLosses = Common.load_integer_list(modelLoadPath / "gLosses.list")
	print(f'loaded from {modelLoadPath}')
	
	runName = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	currentRunPath = runDataPath / runName
	currentModelsPath = currentRunPath / modelsPath 
	intermediateDataPath = currentRunPath / intermediatePath
	currentModelsPath.mkdir(parents=True, exist_ok=True)
	intermediateDataPath.mkdir(parents=True, exist_ok=True)

	transform=transforms.Compose([
		transforms.Resize(imageSize),
		transforms.CenterCrop(imageSize),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	dataset = torchvision.datasets.ImageFolder(root=unalignDataSetPath, transform=transform)

	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
											shuffle=False, num_workers=2, pin_memory=True)

	discriminatorCriterion = nn.BCEWithLogitsLoss()

	lr = 0.0002 #😫
	beta1 = 0.5 #momentumCoef
	beta2 = 0.999 #decayRate
	### m = beta1*m + (1-beta1)*dx
	### cache = beta2*cache + (1-beta2)*(dx**2)
	### x += - learning_rate * m / (np.sqrt(cache) + eps)
	discriminatorOpt = torch.optim.Adam(discriminator.parameters(), lr=lr * 0.5, betas=(beta1, beta2))
	generatorOpt = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
	encoderOpt = torch.optim.Adam(encoder.parameters(), lr = lr, betas=(beta1, beta2))

	for epoch in range(300):
		for i, data in enumerate(dataloader):
			inputs = data[0].to(device)
			b_size = inputs.size(0)
			
			realLabels = torch.full((b_size,1), 1, dtype=torch.float32, device=device, requires_grad=False)
			fakeLabels = torch.full((b_size,1), 0, dtype=torch.float32, device=device, requires_grad=False)
			noise = torch.randn(b_size, latentSize, device=device, requires_grad=False)

			## Generation
			#encoder.zero_grad()
			#generator.zero_grad()
			#encoded = encoder(inputs)
			#decoded = generator(encoded)
			generated = generator(noise)

			## Discriminator optimization
			discriminator.zero_grad()
			## Real error
			realDiscriminatorScore = discriminator(inputs)
			realDiscriminatorError = discriminatorCriterion(realDiscriminatorScore, realLabels)
			## Generated error
			generatedDiscriminatorScore = discriminator(generated.detach())
			generatedDiscriminatorScoreError = discriminatorCriterion(generatedDiscriminatorScore, fakeLabels)
			## Decoded error
			#decodedDiscriminatorScore = discriminator(decoded.detach())
			#decodedDiscriminatorScoreError = discriminatorCriterion(decodedDiscriminatorScore, fakeLabels)
			## Whole error
			#discriminatorError = (realDiscriminatorError + (generatedDiscriminatorScoreError + decodedDiscriminatorScoreError) / 2) / 2
			discriminatorError = (realDiscriminatorError + generatedDiscriminatorScoreError) / 2
			discriminatorError *= 1
			#discriminator.zero_grad()
			discriminatorError.backward(retain_graph=True)
			discriminatorOpt.step()


			## Generator optimization
			## Generated error
			generator.zero_grad()
			generatedDiscriminatorScore2 = discriminator(generated)
			generatedDiscriminatorScore2Error = discriminatorCriterion(generatedDiscriminatorScore2, realLabels)
			### Decoded error
			##decodedDiscriminatorScore2 = discriminator(decoded)
			##decodedDiscriminatorScore2Error = discriminatorCriterion(decodedDiscriminatorScore2, realLabels)
			### Whole error
			#generatorError = (generatedDiscriminatorScore2Error + decodedDiscriminatorScore2Error) / 2
			generatorError = generatedDiscriminatorScore2Error
			generatorError *= 1
			generator.zero_grad()
			generatorError.backward(retain_graph=True)
			generatorOpt.step()

			#encoderOpt.zero_grad()

			## Encoder-Decoder optimization
			#cycleConsistentError = torch.abs(torch.square(inputs - decoded)).sum()
			#cycleConsistentError.backward(retain_graph=True)

			#encoderOpt.step()

			dLosses.append(discriminatorError.item())
			#gLosses.append(generatorError.item())

			if i % 100 == 99:
				print(f'[epoch - {epoch}, {i}/{len(dataloader)}]')
				#print(f'cycle consistent error: {cycleConsistentError.item():>10.3f}')
				print(f'discriminator real (m/e): {torch.nn.functional.sigmoid(realDiscriminatorScore).mean():.6f} / {realDiscriminatorError:.6f}')
				print(f'discriminator generated (m/e): {torch.nn.functional.sigmoid(generatedDiscriminatorScore).mean():.6f} / {generatedDiscriminatorScoreError:.6f}')
				print(f'discriminator generated 2 (m/e): {torch.nn.functional.sigmoid(generatedDiscriminatorScore2).mean():.6f} / {generatedDiscriminatorScore2Error:.6f}')
				#print(f'discriminator decoded (m/e): {decodedDiscriminatorScore.mean():.6f} / {decodedDiscriminatorScoreError:.6f}')
				#print(f'discriminator decoded 2 (m/e): {decodedDiscriminatorScore2.mean():.6f} / {decodedDiscriminatorScore2Error:.6f}')

				with torch.no_grad():
					def saveImages(type, images):
						imagesPath = intermediateDataPath / f'{epoch}_{i}_{type}.png' 
						imageCollection = torchvision.utils.make_grid(images, 4)
						torchvision.utils.save_image(imageCollection, imagesPath)
						print(f'epoch {epoch} is over, intermediate images are stored as \'{imagesPath}\'')
		
					randomImages = generator(torch.randn(8, latentSize, device=device)).detach().cpu() / 2 + 0.5
					saveImages('rand', randomImages)
					#autoencoderImages = torch.cat((inputs[:8].cpu(), decoded[:8].detach().cpu()), 0) / 2 +0.5
					#saveImages('auto', autoencoderImages)

		if epoch % 5 == 0:
			Common.save_integer_list(dLosses, currentModelsPath / "dLosses.list")
			Common.save_integer_list(gLosses, currentModelsPath / "gLosses.list")
			print(f'losses safed to {currentModelsPath}')
			torch.save(discriminator.state_dict(), currentModelsPath / 'discriminator.pth')
			torch.save(generator.state_dict(), currentModelsPath / 'generator.pth')
			print(f'models safed to {currentModelsPath}')

if __name__ == '__main__':
	runDataPath.mkdir(parents=True, exist_ok=True)

	device = torch.device("cuda:0")

	discriminator = Discriminator()
	encoder = Encoder()
	generator = Generator()

	gLosses = []
	dLosses = []

	train = 0
	
	if train == 0:
		learn(discriminator, encoder, generator, dLosses, gLosses)
	elif train == 1:
		summary(discriminator, input_size=(batch_size, 3, imageSize, imageSize))
		summary(generator, input_size=(batch_size, latentSize))
	elif train == 2:
		Common.legend(Path("/home/rrasulov/nntest"), modelLoadPath, dLosses, gLosses)
	else:
		Common.showupGan(Path("/home/rrasulov/nntest"), modelLoadPath, generator, latentSize)
		#Common.showupCycleMyData(Path("/home/rrasulov/nntest"), Path("/home/rrasulov/nntest"), modelLoadPath, generator)
