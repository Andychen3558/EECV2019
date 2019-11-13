import numpy as np
import cv2
import sys
import glob
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm

personNum = 40
imageNum = 10
img_shape = (56, 46)

# def viewImage(img):
# 	cv2.imshow('Display', img)
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()

def process(M): 
	M -= np.min(M)
	M /= np.max(M)
	M = (M * 255).astype(np.uint8)
	return M

def read_data(path):
	data = np.zeros(shape=(personNum, imageNum, img_shape[0]*img_shape[1]))
	for file in glob.glob(path + '/*.png'):
		img = cv2.imread(file, 0)
		file = file.split('/')[-1].split('_')
		personInd = int(file[0]) - 1
		imageInd = int(file[1].split('.')[0]) - 1
		data[personInd][imageInd] = img.flatten().astype(np.float64)
	## Data partition
	train_X = data[:, :7]
	shape = train_X.shape
	train_X = train_X.reshape(shape[0]*shape[1], shape[2])
	train_y = np.array([i//7 for i in range(shape[0]*shape[1])])
	test_X = data[:, 7:]
	shape = test_X.shape
	test_X = test_X.reshape(shape[0]*shape[1], shape[2])
	test_y = np.array([i//3 for i in range(shape[0]*shape[1])])
	return data, train_X, train_y, test_X, test_y

class PCA():
	def __init__(self, data, n):
		self.n = n
		self.data = data
		# Calculate mean && normalize
		self.mean = np.mean(self.data, axis=0)
		self.data -= self.mean
	def getMean(self):
		return self.mean.astype(np.uint8)
	def getPCA(self, use_gramMatrix):
		# Eigen decomposition
		if not use_gramMatrix:
			u, s, v = np.linalg.svd(self.data.T, full_matrices=False)
			return u
		else:
			tmp = self.data.dot(self.data.T)
			u, s, v = np.linalg.svd(tmp.T, full_matrices=False)
			u = self.data.T.dot(u)
			for i in range(u.shape[1]):
				u[:, i] = u[:, i] / np.linalg.norm(u[:, i])
			return u
	def reduce_dimension(self, img):
		img -= self.mean
		u = self.getPCA(use_gramMatrix=False)
		weight = img.dot(u[:, :self.n])
		return weight
	def reconstruct(self, img, use_gramMatrix):
		img -= self.mean
		u = self.getPCA(use_gramMatrix=use_gramMatrix)
		weight = img.dot(u[:, :self.n])
		reconstructed = weight.dot(u[:,0:self.n].T) + self.mean
		return reconstructed.astype(np.uint8)

## Load data
data, train_X, train_y, test_X, test_y = read_data(sys.argv[1])
output_path = sys.argv[2]

def problem3a_1():
	print('[Plot mean face and first 5 eigenfaces]')
	pca = PCA(train_X.copy(), 5)
	meanface = pca.getMean()
	cv2.imwrite(output_path + '/meanface.png', meanface.reshape(img_shape))
	eigen_vecs = pca.getPCA(use_gramMatrix=False)
	for i in range(5):
		eigenface = process(eigen_vecs[:, i])
		cv2.imwrite(output_path + '/eigenface_' + str(i+1) + '.png', eigenface.reshape(img_shape))

def problem3a_2():
	print('[Reconstruct person8_image6]')
	N = [5, 50, 150, len(train_X)-1]
	for n in N:
		pca = PCA(train_X.copy(), n)
		reconstructd_8_6 = pca.reconstruct(data[7][5].copy(), use_gramMatrix=False)
		## Calculate error
		mse = ((reconstructd_8_6 - data[7][5].astype(np.uint8))**2).mean()
		print(mse)
		cv2.imwrite(output_path + '/8_6_' + str(n) + '.png', reconstructd_8_6.reshape(img_shape))

def problem3a_4():
	print('[Reconstruct person8_image6 using Gram Matrix]')
	N = [5, 50, 150, len(train_X)-1]
	for n in N:
		pca = PCA(train_X.copy(), n)
		reconstructd_8_6 = pca.reconstruct(data[7][5].copy(), use_gramMatrix=True)
		## Calculate error
		mse = ((reconstructd_8_6 - data[7][5].astype(np.uint8))**2).mean()
		# print(mse)
		cv2.imwrite(output_path + '/8_6_' + str(n) + '_gram.png', reconstructd_8_6.reshape(img_shape))

def problem3a_3():
	print('[Reduce dim of testing data and visiulization with TSNE]')
	dim = 100
	pca = PCA(train_X.copy(), dim)
	reduced_test = []
	for i in range(len(test_X)):
		tmp = pca.reduce_dimension(test_X[i].copy())
		reduced_test.append(tmp)
	reduced_test = np.array(reduced_test)
	# T-SNE dimension reduction
	embedded = TSNE(n_components=2).fit_transform(reduced_test)
	x_data, y_data = embedded[:, 0], embedded[:, 1]
	# plot
	fig = plt.figure(figsize=(8, 8))
	ax = fig.add_subplot(111)
	cmap = cm.rainbow(np.linspace(0.0, 1.0, 40))
	np.random.shuffle(cmap)
	colors = [cmap[i//3] for i in range(len(embedded))]
	labels = [i for i in range(40)]
	index = [i for i in range(len(embedded))]
	for i, x, y, c in zip(index, x_data, y_data, colors):
		if i%3 == 0:
			ax.scatter(x, y, marker='+', color=c, label=labels[i//3]+1)
		else:
			ax.scatter(x, y, marker='+', color=c)
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
	ax.legend(bbox_to_anchor=(1.03, 0.7), ncol=2)
	plt.savefig('test_distribution')

def problem3b():
	## KNN classifier
	print('[KNN]...')
	K, N = [1, 3, 5], [3, 10, 39]
	best_score, best_k, best_n = 0, 0, 0
	for n in N:
		pca = PCA(train_X.copy(), n)
		tmp_train = []
		for i in range(len(train_X)):
			tmp_train.append(pca.reduce_dimension(train_X[i].copy()))
		tmp_train = np.array(tmp_train)
		# Apply KNN classifier to get best parameters
		for k in K:
			knn = KNeighborsClassifier(n_neighbors=k)
			scores = cross_val_score(knn, tmp_train, train_y, cv=3, scoring='accuracy')
			print(scores.mean())
			if scores.mean() > best_score:
				best_score = scores.mean()
				best_n = n
				best_k = k
	print('==> best_k, best_n: ({:d}, {:d})'.format(best_k, best_n))

	# Apply on testing set
	pca = PCA(test_X.copy(), best_n)
	reduced_test = []
	for i in range(len(test_X)):
		reduced_test.append(pca.reduce_dimension(test_X[i].copy()))
	reduced_test = np.array(reduced_test)
	knn = KNeighborsClassifier(n_neighbors=best_k)
	knn.fit(train_X, train_y)
	predicted = knn.predict(test_X)
	accuracy = metrics.accuracy_score(test_y, predicted)
	print('==> accuracy: {:.3f}'.format(accuracy))

def main():
	## PCA on the training data
	problem3a_1()
	## Processing on person8_image6
	problem3a_2()
	## Processing on testing set with t-sne
	problem3a_3()
	## Gram Matrix trick for PCA
	problem3a_4()

	## KNN
	problem3b()

if __name__ == '__main__':
	main()