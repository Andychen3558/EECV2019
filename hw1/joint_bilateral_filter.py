import numpy as np
import cv2


class Joint_bilateral_filter(object):
	def __init__(self, sigma_s, sigma_r, border_type='reflect'):
		
		self.border_type = border_type
		self.sigma_r = sigma_r
		self.sigma_s = sigma_s

	def joint_bilateral_filter(self, img, guidance):
		r = 3 * self.sigma_s
		h, w = img.shape[:2]
		output = np.zeros_like(img)

		## image padding
		img = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_REFLECT)
		guidance = cv2.copyMakeBorder(guidance, r, r, r, r, cv2.BORDER_REFLECT)
		
		## spacial filter
		x_coor, y_coor = np.meshgrid(np.arange(2 * r + 1) - r, np.arange(2 * r + 1) - r)
		kernel_s = np.exp(-(x_coor**2 + y_coor**2) / (2*self.sigma_s**2))

		## Joint bilateral filter
		guidance = guidance / 255
		output = output.astype(np.float64)
		if guidance.ndim == 3:
			for y in range(r, r+h):
				for x in range(r, r+w):
					kernel_r = np.sum((guidance[y+y_coor, x+x_coor] - guidance[y, x])**2, axis=2)
					kernel_r = np.exp(-(kernel_r) / (2*self.sigma_r**2))
					kernel = kernel_s * kernel_r
					weight = np.sum(kernel)
					for channel in range(3):
						output[y-r, x-r, channel] = np.sum(kernel * img[y+y_coor, x+x_coor, channel]) / weight
					# print(output[y-r][x-r])
		elif guidance.ndim == 2:
			for y in range(r, r+h):
				for x in range(r, r+w):
					kernel_r = (guidance[y+y_coor, x+x_coor] - guidance[y, x])**2
					kernel_r = np.exp(-(kernel_r) / (2*self.sigma_r**2))
					kernel = kernel_s * kernel_r
					weight = np.sum(kernel)
					for channel in range(3):
						output[y-r, x-r, channel] = np.sum(kernel * img[y+y_coor, x+x_coor, channel]) / weight
					# print(output[y-r][x-r])
		return output


