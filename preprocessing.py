from pathlib import Path
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import cv2


class Preprocessing:

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





if __name__ == '__main__':
	test_array = np.zeros((512, 512))
	pp = Preprocessing()
	result = pp.resize(test_array, (300, 300))
