import numpy as np
import cv2
import sys
import time

from joint_bilateral_filter import Joint_bilateral_filter

def main():
	## load image
	img = cv2.imread(sys.argv[1])
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	## conventional rgb2gray conversion
	w = np.array([0.299, 0.587, 0.114])
	img_gray = np.dot(img_rgb, w)
	filename = sys.argv[1].split('/')[2].split('.')[0]
	cv2.imwrite(filename + '_gray.png', img_gray)

	## init parameters
	sigma_s = [1, 2, 3]
	sigma_r = [0.05, 0.1, 0.2]

	## number of votes for each set of parameters
	N = 10
	votes = np.zeros(shape=(N+1, N+1))

	## Joint_bilateral_filter
	for s in sigma_s:
		for r in sigma_r:
			print('sigma_s:' + str(s) + '  sigma_r:' + str(r))
			## create JBF class
			JBF = Joint_bilateral_filter(s, r, border_type='reflect')
			## bilateral_filter
			bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)

			error = np.full((11, 11), float('inf')).astype(np.float64)

			for i in range(N+1):
				for j in range(N+1-i):
					w_cur = np.array([i/10, j/10, (10-i-j)/10])
					## convert to grayscale
					img_gray = np.dot(img_rgb, w_cur)

					jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray).astype(np.uint8)

					## cal error and update votes
					err = np.sum(np.abs(jbf_out - bf_out))
					# print(err)
					error[i][j] = err
			# error = np.random.randint(0, 2000, size=(11, 11))
			print(error)

			## find local minima with 6 matrices
			compare = np.zeros(shape=(6, 11, 11))
			### up point
			up = np.roll(error, 1, axis=0).astype(np.float64)
			up[0, :] = float('inf')
			compare[0] = up
			### down
			down = np.roll(error, -1, axis=0).astype(np.float64)
			down[N, :] = float('inf')
			compare[1] = down
			### left
			left = np.roll(error, 1, axis=1).astype(np.float64)
			left[:, 0] = float('inf')
			compare[2] = left
			### right
			right = np.roll(error, -1, axis=1).astype(np.float64)
			right[:, N] = float('inf')
			compare[3] = right
			### upper-right
			upperRight = np.roll(np.roll(error, 1, axis=0).astype(np.float64), -1, axis=1).astype(np.float64)
			upperRight[0, :] = float('inf')
			upperRight[:, N] = float('inf')
			compare[4] = upperRight
			### lower-left
			lowerLeft = np.roll(np.roll(error, -1, axis=0).astype(np.float64), 1, axis=1).astype(np.float64)
			lowerLeft[N, :] = float('inf')
			lowerLeft[:, 0] = float('inf')
			compare[5] = lowerLeft

			### get result
			result = np.full((N+1, N+1), True)
			for i in range(len(compare)):
				result &= (error < compare[i])
			votes[result] += 1
			print(votes)

	ans = np.dstack(np.unravel_index(np.argsort(votes.ravel()), (N+1, N+1)))[0][-3:][::-1]
	print(filename)
	for i in range(3):
		w = np.array([ans[i][0] / 10, ans[i][1] / 10, (10-ans[i][0]-ans[i][1]) / 10])
		print(w)
		img_gray = np.dot(img_rgb, w)
		cv2.imwrite(filename + '_y' + str(i+1) + '.png', img_gray)


if __name__ == '__main__':
	main()