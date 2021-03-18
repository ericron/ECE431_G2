from pathlib import Path
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy as np
import numpy.ma as ma


class Preprocessing:
	def __init__(self, dcmImage_ID):
		self.dcmImage_ID = dcmImage_ID

	def resize(self, array, new_size):
		print("Original Size:", array.shape)
		plt.imshow(array, cmap='gray')
		plt.show()
		# width = array.shape[1]
		# height = array.shape[0]
		# interpolation types: INTER_NEAREST, INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTERLANCZOS4
		resized = cv2.resize(array, new_size, interpolation=cv2.INTER_LINEAR)
		print("New Size:", resized.shape)
		plt.imshow(resized, cmap='gray')
		plt.show()
		return resized

	def make_mask(self, imageArray, display=False):
		imghound = self.dcmImage_ID.scale_to_hu(imageArray)
		kernel = np.ones((8, 8), np.uint8)
		erosion = cv2.erode(imghound, kernel, iterations=2)
		threshhold = np.where(erosion < -500, 0, 1)
		mask = np.where(threshhold < 0.5, 1, 0)
		final = ma.masked_array(imghound, mask)
		if (display):
			fig, ax = plt.subplots(3, 2, figsize=[12, 12])
			ax[0, 0].set_title("Original")
			ax[0, 0].imshow(imghound, cmap='gray')
			ax[0, 0].axis('off')
			ax[0, 1].set_title("Erosion")
			ax[0, 1].imshow(erosion, cmap='gray')
			ax[0, 1].axis('off')
			ax[1, 0].set_title("Threshold")
			ax[1, 0].imshow(threshhold, cmap='gray')
			ax[1, 0].axis('off')
			ax[1, 1].set_title("Mask")
			ax[1, 1].imshow(mask, cmap='gray')
			ax[1, 1].axis('off')
			ax[2, 0].set_title("Masked Image")
			ax[2, 0].imshow(final, cmap='gray')
			ax[2, 0].axis('off')
			plt.show()
		return final


if __name__ == '__main__':
	test_array = np.zeros((512, 512))
	pp = Preprocessing(None)
	result = pp.resize(test_array, (300, 300))
