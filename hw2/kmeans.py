import numpy as np
import cv2
import sys
import glob
from sklearn.manifold import TSNE
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pca import PCA

personNum = 40
imageNum = 10
img_shape = (56, 46)

output_path = sys.argv[2]

def read_data(path):
	data = np.zeros(shape=(personNum, imageNum, img_shape[0]*img_shape[1]))
	for file in glob.glob(path + '/*.png'):
		img = cv2.imread(file, 0)
		file = file.split('/')[1].split('_')
		personInd = int(file[0]) - 1
		imageInd = int(file[1].split('.')[0]) - 1
		data[personInd][imageInd] = img.flatten().astype(np.float64)
	train_X = data[:10, :7]
	shape = train_X.shape
	train_X = train_X.reshape(shape[0]*shape[1], shape[2])
	train_y = np.array([i//7 for i in range(shape[0]*shape[1])])
	return train_X, train_y

class K_Means():
	def __init__(self, k=10, tol=0.0001, max_iter=15):
		self.k = k
		self.tol = tol
		self.max_iter = max_iter
	def weighted_Euclidean_distance(self, x, y):
		w = np.array([0.6, 0.4, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
		return np.sqrt(np.dot(w, (x-y)**2))
	def fit(self, data, labels):
		## Init data
		self.centroids = np.zeros(shape=(self.k, len(data[0])))
		for i in range(self.k):
			self.centroids[i] = data[i]
		# Init color map
		cmap = cm.rainbow(np.linspace(0.0, 1.0, self.k))
		np.random.shuffle(cmap)

		## PCA to project data on 2D space
		pca = PCA(data.copy(), 2)

		## Iteration of K-means algo
		for _ in range(self.max_iter):
			self.clf = [[] for _ in range(self.k)]

			# Assign all data to closest center
			for i in range(len(data)):
				distances = [self.weighted_Euclidean_distance(data[i], self.centroids[j]) for j in range(len(self.centroids))]
				class_ = distances.index(min(distances))
				self.clf[class_].append([data[i], labels[i]])

			old_centroids = self.centroids.copy()

			# Update centroids
			for i in range(len(self.clf)):
				tmp_data = [self.clf[i][j][0] for j in range(len(self.clf[i]))]
				self.centroids[i] = np.mean(tmp_data, axis=0)
				
			# End if converged
			converged = True
			for i in range(len(self.centroids)):
				if np.sum((self.centroids[i] - old_centroids[i]) / old_centroids[i]) > self.tol:
					print('Class: %d, error: %f' %(i, np.sum((self.centroids[i] - old_centroids[i]) / old_centroids[i])))
					converged = False
			if converged:
				break
			print('-----')

			## Reduce dimension to 2-dim and visualize
			for i in range(len(self.centroids)):
				curr = pca.reduce_dimension(self.centroids[i].copy()).tolist()
				plt.scatter(curr[0], curr[1], marker='o', color=cmap[i])
			for i in range(len(self.clf)):
				for j in range(len(self.clf[i])):
					curr = pca.reduce_dimension(self.clf[i][j][0].copy()).tolist()
					plt.scatter(curr[0], curr[1], marker='x', color=cmap[i])
			# plt.savefig('%s/kmeans_%d.png' %(output_path, _))
		plt.savefig(output_path)
		return self.clf


def main():
	## Load data
	train_X, train_y = read_data(sys.argv[1])
	N = len(train_X)
	indices = np.arange(train_X.shape[0])
	np.random.shuffle(indices)
	train_X = train_X[indices]
	train_y = train_y[indices]

	## Reduce dimension to 10-dim
	dim = 10
	pca = PCA(train_X.copy(), dim)
	reduced_data = []
	for i in range(N):
		tmp = pca.reduce_dimension(train_X[i].copy())
		reduced_data.append(tmp)
	reduced_data = np.array(reduced_data)

	## Apply K-means algo
	clf = K_Means()
	res = clf.fit(reduced_data, train_y)
	
	## Compare the result with ground truth
	correct_cnt = 0
	for i in range(len(res)):
		labels = [res[i][j][1] for j in range(len(res[i]))]
		if not set(labels):
			continue
		class_ = max(set(labels), key=labels.count)
		correct_cnt += labels.count(class_)
	print(correct_cnt / N)

if __name__ == '__main__':
	main()