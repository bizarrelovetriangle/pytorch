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
from CelebADataset import CelebADataset
import pandas as pd
from torch.nn.utils import spectral_norm

# Root directory for dataset
dataroot = "data/celeba"

chanels = 3

disFeaturesCount = 32
genFeaturesCount = 32

latentSize = 124

imageSize = 128

batch_size = 128

baseTrainingPath = Path("/home/rrasulov")
#baseTrainingPath = Path("E:/NNTrainDirection")

attributeFactor = 0.2
attributeFilters = ['Male', 'Big_Lips', 'Chubby', 'Attractive', 'Young']
attributesSize = len(attributeFilters)
attributeLabelsPath = baseTrainingPath / Path("training_data/list_attr_celeba.txt")

alignDataSetPath = baseTrainingPath / Path("training_data/align_dataset/img_align_celeba")
unalignDataSetPath = baseTrainingPath / Path("training_data/unalign_dataset/img_celeba")
#modelsPath = pathlib.Path(__file__).parent.resolve()
runDataPath = baseTrainingPath / Path("run_data")
modelsPath = "models"
intermediatePath = "intermediete_images"
modelLoadPath = baseTrainingPath / Path("run_data/20250828-104044/models")

class Discriminator(nn.Module):
	def __init__(self, gn_groups=8):
		super(Discriminator, self).__init__()
		g = gn_groups

		self.main = nn.Sequential(
			spectral_norm(nn.Conv2d(3, disFeaturesCount, kernel_size=3, stride=1, padding=1, bias=False)),
			nn.LeakyReLU(0.2, inplace=True),

			spectral_norm(nn.Conv2d(disFeaturesCount, disFeaturesCount, kernel_size=4, stride=2, padding=1, bias=False)),
			nn.GroupNorm(g, disFeaturesCount),
			nn.LeakyReLU(0.2, inplace=True),

			spectral_norm(nn.Conv2d(disFeaturesCount, disFeaturesCount * 2, kernel_size=4, stride=2, padding=1, bias=False)),
			nn.GroupNorm(g, disFeaturesCount * 2),
			nn.LeakyReLU(0.2, inplace=True),

			spectral_norm(nn.Conv2d(disFeaturesCount * 2, disFeaturesCount * 4, kernel_size=4, stride=2, padding=1, bias=False)),
			nn.GroupNorm(g, disFeaturesCount * 4),
			nn.LeakyReLU(0.2, inplace=True),

			spectral_norm(nn.Conv2d(disFeaturesCount * 4, disFeaturesCount * 8, kernel_size=4, stride=2, padding=1, bias=False)),
			nn.GroupNorm(g, disFeaturesCount * 8),
			nn.LeakyReLU(0.2, inplace=True),

			spectral_norm(nn.Conv2d(disFeaturesCount * 8, disFeaturesCount * 16, kernel_size=4, stride=2, padding=1, bias=False)),
			nn.GroupNorm(g, disFeaturesCount * 16),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.likelihoodFinal = nn.Sequential(
			nn.Flatten(),
			spectral_norm(nn.Linear(disFeaturesCount * 16 * 4 * 4, 1, bias=True)),
		)

		self.attributesFinal = nn.Sequential(
			nn.Flatten(),
			spectral_norm(nn.Linear(disFeaturesCount * 16 * 4 * 4, attributesSize, bias=True)),
		)

	def forward(self, images):
		B = images.size(0)
		x = self.main(images)
		likelihood = self.likelihoodFinal(x)
		attributes = self.attributesFinal(x)
		return likelihood, attributes

class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.main = nn.Sequential(
			nn.LazyConvTranspose2d(genFeaturesCount * 8, kernel_size=4, stride=2, padding=1, bias=False),
			nn.LazyBatchNorm2d(),
			nn.ReLU(inplace=True),

			nn.LazyConvTranspose2d(genFeaturesCount * 4, kernel_size=4, stride=2, padding=1, bias=False),
			nn.LazyBatchNorm2d(),
			nn.ReLU(inplace=True),

			Common.Interpolate(size=(32, 32), mode='nearest'),
			nn.LazyConv2d(genFeaturesCount * 2, kernel_size=3, stride=1, padding=1, bias=False),
			nn.LazyBatchNorm2d(),
			nn.ReLU(inplace=True),

			Common.Interpolate(size=(64, 64), mode='nearest'),
			nn.LazyConv2d(genFeaturesCount, kernel_size=3, stride=1, padding=1, bias=False),
			nn.LazyBatchNorm2d(),
			nn.ReLU(inplace=True),
			
			Common.Interpolate(size=(128, 128), mode='nearest'),
			nn.LazyConv2d(genFeaturesCount, kernel_size=3, stride=1, padding=1, bias=False),
			nn.LazyBatchNorm2d(),
			nn.ReLU(inplace=True),

			nn.LazyConvTranspose2d(chanels, kernel_size=3, stride=1, padding=1, bias=True),
			nn.Tanh()
		)

		self.project = nn.Sequential(
			nn.Linear(latentSize + attributesSize, genFeaturesCount * 16 * 4 * 4, bias=False),
			nn.LazyBatchNorm1d(),
		)

	def forward(self, noise, attributes):
		B = noise.size(0)
		input = torch.cat((noise, attributes), 1)
		x = self.project(input).view(B, genFeaturesCount * 16, 4, 4)
		return self.main(x)
	
def learn(discriminator, generator, dLosses, gLosses):
	discriminator.to(device)
	discriminator.apply(Common.weights_init)

	generator.to(device)
	generator.apply(Common.weights_init)
	
	discriminator.load_state_dict(torch.load(modelLoadPath / 'discriminator.pth', weights_only=True), strict=False)
	generator.load_state_dict(torch.load(modelLoadPath / 'generator.pth', weights_only=True), strict=False)
	#dLosses = Common.load_integer_list(modelLoadPath / "dLosses.list")
	#gLosses = Common.load_integer_list(modelLoadPath / "gLosses.list")
	print(f'loaded from {modelLoadPath}')

	# Male Big_Lips Chubby Attractive Young
	attributeLabels = pd.read_csv(attributeLabelsPath, sep=r'\s+', header=1)
	attributeLabels = attributeLabels[attributeFilters]

	runData = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	currentRunPath = runDataPath / (runData + '_SupervisedGan')
	currentModelsPath = currentRunPath / modelsPath 
	intermediateDataPath = currentRunPath / intermediatePath
	currentModelsPath.mkdir(parents=True, exist_ok=True)
	intermediateDataPath.mkdir(parents=True, exist_ok=True)

	transform=transforms.Compose([
		transforms.Resize(imageSize),
		transforms.CenterCrop(imageSize),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	dataset = CelebADataset(alignDataSetPath, attributeLabels, transform=transform)

	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
											 shuffle=True, num_workers=2, pin_memory=True)

	likelihoodCriterion = nn.BCEWithLogitsLoss()
	attributesCriterion = nn.BCEWithLogitsLoss()

	lr = 0.0002 #😫
	beta1 = 0.5 #momentumCoef
	beta2 = 0.999 #decayRate
	### m = beta1*m + (1-beta1)*dx
	### cache = beta2*cache + (1-beta2)*(dx**2)
	### x += - learning_rate * m / (np.sqrt(cache) + eps)
	discriminatorOpt = torch.optim.Adam(discriminator.parameters(), lr=lr * 0.5, betas=(beta1, beta2))
	generatorOpt = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))

	for epoch in range(300):
		for i, data in enumerate(dataloader):
			images = data[0].to(device)
			attributes = data[1].to(device)
			attributes01 = (attributes + 1) / 2
			b_size = images.size(0)

			realLabels = torch.full((b_size,1), 1, dtype=torch.float32, device=device, requires_grad=False)
			fakeLabels = torch.full((b_size,1), 0, dtype=torch.float32, device=device, requires_grad=False)
			noise = torch.randn(b_size, latentSize, device=device, requires_grad=False)
			generated = generator(noise, attributes)

			# Discriminator likelihood error
			discriminator.zero_grad()
			realDiscriminatorScore, predictedRealAttributes = discriminator(images)
			realDiscriminatorError = likelihoodCriterion(realDiscriminatorScore, realLabels)
			generatedDiscriminatorScore, _ = discriminator(generated.detach())
			generatedDiscriminatorScoreError = likelihoodCriterion(generatedDiscriminatorScore, fakeLabels)

			discriminatorLikelihoodError = (realDiscriminatorError + generatedDiscriminatorScoreError) / 2
			discriminatorLikelihoodError.backward(retain_graph=True)

			# Discriminator attributes error
			discriminatorAttributesError = attributeFactor * attributesCriterion(predictedRealAttributes, attributes01)
			discriminatorAttributesError.backward(retain_graph=True)

			discriminatorOpt.step()


			## Generator likelihood optimization
			generator.zero_grad()
			generatedDiscriminatorScore2, predictedFakeAttributes = discriminator(generated)
			generatedDiscriminatorScore2Error = likelihoodCriterion(generatedDiscriminatorScore2, realLabels)
			
			generatorError = generatedDiscriminatorScore2Error
			generatorError.backward(retain_graph=True)

			# Generator attributes error
			generatorAttributesError = attributeFactor * attributesCriterion(predictedFakeAttributes, attributes01)
			generatorAttributesError.backward(retain_graph=True)

			generatorOpt.step()


			dLosses.append((discriminatorLikelihoodError).item())
			gLosses.append((generatorError).item())

			if i % 100 == 99:
				print(f'[epoch - {epoch}, {i}/{len(dataloader)}]')
				print(f'discriminator real (m/e): {torch.nn.functional.sigmoid(realDiscriminatorScore).mean():.6f} / {realDiscriminatorError:.6f}')
				print(f'discriminator generated (m/e): {torch.nn.functional.sigmoid(generatedDiscriminatorScore).mean():.6f} / {generatedDiscriminatorScoreError:.6f}')
				print(f'discriminator generated 2 (m/e): {torch.nn.functional.sigmoid(generatedDiscriminatorScore2).mean():.6f} / {generatedDiscriminatorScore2Error:.6f}')
				print(f'discriminatorAttributesError: {discriminatorAttributesError:.6f}')
				print(f'generatorAttributesError: {generatorAttributesError:.6f}')

				with torch.no_grad():
					def saveImages(type, images):
						imagesPath = intermediateDataPath / f'{epoch}_{i}_{type}.png'
						imageCollection = torchvision.utils.make_grid(images, 4)
						torchvision.utils.save_image(imageCollection, imagesPath)
						print(f'intermediate images are stored as \'{imagesPath}\'')
		
					stdRandomImages = generator(
						torch.randn(8, latentSize, device=device),
						attributes[:8]).detach().cpu() / 2 + 0.5
					saveImages('stdrand', stdRandomImages)
					
					fullyRandomImages = generator(
						torch.randn(8, latentSize, device=device),
						torch.randn(8, attributesSize, device=device)).detach().cpu() / 2 + 0.5
					saveImages('fullyrand', fullyRandomImages)

					#attributeFilters = ['Male', 'Big_Lips', 'Chubby', 'Attractive', 'Young']
					attractiveAttributes = torch.tensor([0., 1., 1., 1., 1.], device=device).repeat(8, 1)
					attractiveRandomImages = generator(
						torch.randn(8, latentSize, device=device),
						attractiveAttributes).detach().cpu() / 2 + 0.5
					saveImages('attractiverand', attractiveRandomImages)

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
	generator = Generator()

	gLosses = []
	dLosses = []

	train = 0

	if train == 0:
		learn(discriminator, generator, dLosses, gLosses)
	elif train == 1:
		summary(discriminator, input_size=(batch_size, 3, imageSize, imageSize))
		summary(generator, input_size=(batch_size, latentSize))
	elif train == 2:
		Common.legend(Path("/home/rrasulov/nntest"), modelLoadPath, dLosses, gLosses)
	else:
		Common.showupGan(Path("/home/rrasulov/nntest"), modelLoadPath, generator, latentSize)
		#Common.showupCycleMyData(Path("/home/rrasulov/nntest"), Path("/home/rrasulov/nntest"), modelLoadPath, generator)
