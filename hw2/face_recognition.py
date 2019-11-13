import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torchvision.models import alexnet
# from torchsummary import summary
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import glob, os
from baseline import get_dataloader

# Specifiy data folder path
folder = sys.argv[1]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		self.cnn = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Dropout(0.25),  # [64, 48, 48]

			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Dropout(0.3),   # [128, 24, 24]

			nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Dropout(0.35),  # [512, 12, 12]

			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Dropout(0.35),  # [512, 6, 6]
		)
		self.fc = nn.Sequential(
			nn.Linear(512*6*6, 1024),
			nn.BatchNorm1d(1024),
			nn.LeakyReLU(0.2),
			nn.Dropout(0.5),
			nn.Linear(1024, 512),
			nn.BatchNorm1d(512),
			nn.LeakyReLU(0.2),
			nn.Dropout(0.5),
			nn.Linear(512, 100)
		)

	def get_feature(self, x):
		out = self.cnn(x)
		return out.view(out.size()[0], -1)

	def forward(self, x):
		out = self.cnn(x)
		out = out.view(out.size()[0], -1)
		out = self.fc(out)
		return out

	def name(self):
		return "ConvNet"


def main():
	## Load data 
	print('[Loading data]...')
	train_loader, val_loader = get_dataloader(folder, 32)
	
	## Init parameters
	num_epoch = 10
	best_acc = 0.0

	Acc_train, Loss_train = [], []
	Acc_val, Loss_val = [], []
	Epoch = [i for i in range(1, 1 + num_epoch)]

	model = ConvNet().to(device)
	# Output model summary
	# summary(model, (3, 96, 96))

	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

	for epoch in range(num_epoch):
		print('[Epoch: {:d}]'.format(epoch))

		# Record the information of correct prediction and loss
		correct_cnt, total_loss, total_cnt = 0, 0, 0

		## Training
		model.train()
		print('[Training]...')
		for batch, (img, label) in enumerate(train_loader, 1):
			# Set the gradients to zero (left by previous iteration)
			optimizer.zero_grad()
			# Put input tensor to GPU if it's available
			img, label = img.to(device), label.to(device)
			# Forward input tensor through your model
			out = model(img)
			# Calculate loss
			loss = criterion(out, label)
			# Compute gradient of each model parameters base on calculated loss
			loss.backward()
			# Update model parameters using optimizer and gradients
			optimizer.step()

			# Calculate the training loss and accuracy of each iteration
			total_loss += loss.item()
			_, pred_label = torch.max(out, 1)
			total_cnt += img.size(0)
			correct_cnt += (pred_label == label).sum().item()

			# Show the training information
			if batch % 5 == 0 or batch == len(train_loader):
				acc = correct_cnt / total_cnt
				ave_loss = total_loss / batch		 
				print ('Training batch index: {}, train loss: {:.6f}, acc: {:.3f}'.format(
					batch, ave_loss, acc))

		# Store acc && loss to plot
		Acc_train.append(acc)
		Loss_train.append(ave_loss)

		# Record the information of correct prediction and loss
		correct_cnt, total_loss, total_cnt = 0, 0, 0

		# Testing
		model.eval()
		with torch.no_grad():
			print('[Evaluating]...')
			for batch, (img, label) in enumerate(val_loader, 1):
				# Put input tensor to GPU if it's available
				img, label = img.to(device), label.to(device)
				# Forward input tensor through your model
				out = model(img)
				# Calculate loss
				loss = criterion(out, label)

				# Calculate the training loss and accuracy of each iteration
				total_loss += loss.item()
				_, pred_label = torch.max(out, 1)
				total_cnt += img.size(0)
				correct_cnt += (pred_label == label).sum().item()

				# Show the training information
				if batch % 5 == 0 or batch == len(val_loader):
					acc = correct_cnt / total_cnt
					ave_loss = total_loss / batch		 
					print ('Validation batch index: {}, valid loss: {:.6f}, acc: {:.3f}'.format(
						batch, ave_loss, acc))
			if acc > best_acc:
				best_acc = acc

		# Store acc && loss to plot
		Acc_val.append(acc)
		Loss_val.append(ave_loss)

	print('==> best_acc: {:.3f}'.format(best_acc))

	# Save trained model
	torch.save(model.state_dict(), './checkpoint/%s_face.pth' % model.name())

	 # Plot Learning Curve
	fig, (acc_curve, loss_curve) = plt.subplots(2, 1)
	fig.subplots_adjust(hspace=0.5)

	## Accuracy curve
	acc_curve.set_title('Accuracy curve')
	acc_curve.plot(Epoch, Acc_train, color='c', label='Training Accuracy')
	acc_curve.plot(Epoch, Acc_val, color='r', label='Validation Accuracy')
	acc_curve.legend(loc='lower right')

	## Loss curve
	loss_curve.set_title('Loss curve')
	loss_curve.plot(Epoch, Loss_train, color='c', label='Training Loss')
	loss_curve.plot(Epoch, Loss_val, color='r', label='Validation Loss')
	loss_curve.legend(loc='upper right')

	## Output the curves
	fig.savefig('curves_faceCNN.png')

if __name__ == "__main__":
	main()