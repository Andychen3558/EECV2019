import sys
import numpy as np
from sklearn import metrics
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import glob, os
from face_recognition import ConvNet
from baseline import visualization

# Specifiy data folder path and output path
folder, output_tsne = sys.argv[1], sys.argv[2]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_dataloader(folder, batch_size):
	# Get data loaders of training set and validation set
	trans = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
	)])
	train_path, test_path = os.path.join(folder,'train'), os.path.join(folder,'valid')
	# Get dataset using pytorch functions
	train_set = ImageFolder(train_path, transform=trans)
	test_set =  ImageFolder(test_path,  transform=trans)
	train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False, num_workers=8)
	test_loader  = torch.utils.data.DataLoader(dataset=test_set,  batch_size=batch_size, shuffle=False, num_workers=8)
	print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
	print ('==>>> total testing batch number: {}'.format(len(test_loader)))
	return train_loader, test_loader

def main():
	## Load validation data
	train_loader, val_loader = get_dataloader(folder, 1)

	## Load CNN model
	model = ConvNet()
	model.load_state_dict(torch.load('./checkpoint/ConvNet_face.pth'))
	model = model.to(device)

	## Get training_data features
	# with torch.no_grad():
	# 	print('[Retriving image features]...')
	# 	for batch, (img, label) in enumerate(train_loader):
	# 		# Put input tensor to GPU if it's available
	# 		img, label = img.to(device), label.to(device)
	# 		# Forward input tensor through your model
	# 		out = model.get_feature(img)
	# 		out = out.cpu().numpy()
	# 		label = label.cpu().numpy()

	# 		if batch == 0:
	# 			data = out
	# 			class_ = label
	# 		elif label == 10:
	# 			break
	# 		else:
	# 			data = np.concatenate((data, out), axis=0)
	# 			class_ = np.concatenate((class_, label), axis=0)

	## Get validation_data features
	with torch.no_grad():
		print('[Retriving image features]...')
		for batch, (img, label) in enumerate(val_loader):
			# Put input tensor to GPU if it's available
			img, label = img.to(device), label.to(device)
			# Forward input tensor through your model
			out = model.get_feature(img)
			out = out.cpu().numpy()
			label = label.cpu().numpy()

			if batch == 0:
				data = out
				class_ = label
			elif label == 10:
				break
			else:
				data = np.concatenate((data, out), axis=0)
				class_ = np.concatenate((class_, label), axis=0)

	## Visualization
	visualization(data, class_, output_tsne)


if __name__ == "__main__":
	main()