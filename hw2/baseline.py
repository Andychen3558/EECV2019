###################################################################################
## Problem 4(b):																 ##
## You should extract image features using pytorch pretrained alexnet and train  ##
## a KNN classifier to perform face recognition as your baseline in this file.   ##
###################################################################################

import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.manifold import TSNE
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from torchvision.models import alexnet
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import glob, os

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
	train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=8)
	test_loader  = torch.utils.data.DataLoader(dataset=test_set,  batch_size=batch_size, shuffle=False, num_workers=8)
	print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
	print ('==>>> total testing batch number: {}'.format(len(test_loader)))
	return train_loader, test_loader

def visualization(data, label, output):
	print('[Visualization]')
	# T-SNE dimension reduction
	embedded = TSNE(n_components=2).fit_transform(data)
	# Init color map
	identities = 10
	cmap = cm.rainbow(np.linspace(0.0, 1.0, identities))
	np.random.shuffle(cmap)
	for i in range(len(embedded)):
		plt.scatter(embedded[i][0], embedded[i][1], marker='o', color=cmap[label[i]])
	plt.savefig(output)

if __name__ == "__main__":
	# TODO
	## Load data 
	print('[Loading data]...')
	train_loader, test_loader = get_dataloader(folder, 32)

	## Extract image features using pytorch pretrained alexnet
	with torch.no_grad():
		# Training set
		print('[Extracting training data]...')
		for batch, (img, label) in enumerate(train_loader):
			extracter = alexnet(pretrained=True).features
			extracter.to(device)
			extracter.eval()
			img = img.to(device)
			features = extracter(img)
			features = features.view(img.size(0), 256, -1)
			features = torch.mean(features, 2)
			features = features.cpu().numpy()
			label = label.cpu().numpy()
			if batch == 0:
				train_X = features.copy()
				train_y = label.copy()
			else:
				train_X = np.concatenate((train_X, features), axis=0)
				train_y = np.concatenate((train_y, label), axis=0)
		print(train_X.shape, train_y.shape)

		# Validation set
		print('[Extracting testing data]...')
		for batch, (img, label) in enumerate(test_loader):
			img = img.to(device)
			features = extracter(img).view(img.size(0), 256, -1)
			features = torch.mean(features, 2)
			features = features.cpu().numpy()
			label = label.cpu().numpy()
			if batch == 0:
				test_X = features.copy()
				test_y = label.copy()
			else:
				test_X = np.concatenate((test_X, features), axis=0)
				test_y = np.concatenate((test_y, label), axis=0)
		print(test_X.shape, test_y.shape)

	# Reduce dimension to n-dim
	print('[Dimension reduction]')
	n = 128
	pca = PCA(n_components=n).fit(train_X)
	train_X = pca.transform(train_X)
	test_X = pca.transform(test_X)

	## Apply KNN classifier
	print('[KNN]')
	k = 3
	knn = KNeighborsClassifier(n_neighbors=k)
	knn.fit(train_X, train_y)
	predicted = knn.predict(test_X)
	accuracy = metrics.accuracy_score(test_y, predicted)
	print('==> accuracy: %f' %(accuracy))

	## Visualization for first 10 identities
	# visualization(train_X[:692], train_y[:692], output_tsne)
	visualization(test_X[:145], test_y[:145], output_tsne)