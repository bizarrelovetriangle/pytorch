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

class Common:
	def save_integer_list(int_list, file_path):
		with open(file_path, 'wb') as f:
			f.write(struct.pack('I', len(int_list)))
			f.write(struct.pack(f'{len(int_list)}f', *int_list))

	def load_integer_list(file_path):
		with open(file_path, 'rb') as f:
			list_length = struct.unpack('I', f.read(4))[0]
			int_list = list(struct.unpack(f'{list_length}f', f.read(4 * list_length)))
		return int_list

	def legend(outputPath, path, dLosses, gLosses):
		dLosses = Common.load_integer_list(path / "dLosses.list")
		gLosses = Common.load_integer_list(path / "gLosses.list")

		plt.figure(figsize=(10,5))
		plt.title("Generator and Discriminator Loss During Training")
		plt.plot(gLosses,label="G")
		plt.plot(dLosses,label="D")
		plt.xlabel("iterations")
		plt.ylabel("Loss")
		plt.legend()
		plt.show()
		plt.savefig(outputPath / "legent.png")

	def imshow(img, unnormalize = True, transpose = True):
		if unnormalize:
			img = img / 2 + 0.5     # unnormalize
		npimg = img.detach().cpu().data.numpy()
		if transpose:
			npimg = np.transpose(npimg, (1, 2, 0))
		plt.axis('off')
		plt.style.use('dark_background')
		plt.imshow(npimg)
		plt.save()

	def weights_init(m):
		classname = m.__class__.__name__
		if classname.find('Conv') != -1:
			nn.init.normal_(m.weight.data, 0.0, 0.02)
		elif classname.find('BatchNorm') != -1:
			nn.init.normal_(m.weight.data, 1.0, 0.02)
			nn.init.constant_(m.bias.data, 0)

	def showupGan(outputPath, loadPath, generator, latentSize):
		generator.load_state_dict(torch.load(loadPath / 'generator.pth', weights_only=True))
		print(f'loaded from {loadPath}')

		output = generator(torch.randn(16, latentSize))
		images = torchvision.utils.make_grid(output.detach().cpu(), 8) / 2 + 0.5
		torchvision.utils.save_image(images, outputPath / "zShowupGanResult.png")

	def showupCycleMyData(imagesPath, outputPath, loadPath, encoder, generator):
		encoder.load_state_dict(torch.load(loadPath / 'encoder.pth', weights_only=True))
		generator.load_state_dict(torch.load(loadPath / 'generator.pth', weights_only=True))
		print(f'loaded from {loadPath}')

		transform=transforms.Compose([
			transforms.Resize((142, 142)),
			transforms.CenterCrop((142, 142)),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		dataset = torchvision.datasets.ImageFolder(root=imagesPath, transform=transform)

		dataloader = torch.utils.data.DataLoader(dataset, batch_size=50,
												shuffle=True, num_workers=2)

		dataIter = next(iter(dataloader))
		inputs = dataIter[0]

		encoded = encoder(inputs)
		output = generator(encoded).detach().cpu()

		images = torchvision.utils.make_grid(output, 8) / 2 + 0.5
		torchvision.utils.save_image(images, outputPath / "zShowupCycleMyDataResult.png")
		print(f'images are stored as \'{outputPath}\'')

	class Interpolate(torch.nn.Module):
		def __init__(self, size, mode):
				super().__init__()
				self.size = size
				self.mode = mode

		def forward(self, input):
			return nn.functional.interpolate(input, size=self.size, mode=self.mode)

	class SpectralNorm(torch.nn.Module):
		def __init__(self):
			super().__init__()
		
		def forward(self, input):
			return torch.nn.utils.spectral_norm(input)

	# def showupCycle():
	#     encoder = Encoder()
	#     decoder = Generator()
	#     encoder.load_state_dict(torch.load(netPath / 'encoder.pth', weights_only=True))
	#     decoder.load_state_dict(torch.load(netPath / 'generator.pth', weights_only=True))

	#     transform=transforms.Compose([
	#         transforms.Resize((imageHeight, imageWidth)),
	#         transforms.CenterCrop((imageHeight, imageWidth)),
	#         transforms.ToTensor(),
	#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	#     dataset = torchvision.datasets.ImageFolder(root=path, transform=transform)

	#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
	#                                             shuffle=True, num_workers=2)
	#     dataIter = next(iter(dataloader))
	#     inputs = dataIter[0]

	#     encoded = encoder(inputs)
	#     output = decoder(encoded)

	#     ba = torch.cat((inputs[:16].cpu(), output[:16].detach().cpu()), 0)
	#     print(ba.shape)
	#     imshow(torchvision.utils.make_grid(ba, 8))



# do I need it?
class SchedulerParams:
	ideal_loss = np.log(4)
	x_min = 0.1 * np.log(4)
	x_max = 0.1 * np.log(4)
	h_min = 0.1
	f_max = 2.0
	
