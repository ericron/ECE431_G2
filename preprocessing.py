from pathlib import Path
import pydicom
import matplotlib.pyplot as plt
import cv2
import numpy as np
import numpy.ma as ma
import pandas as pd


class Preprocessing:
	def __init__(self, dcmImage_ID):
		self.dcmImage_ID = dcmImage_ID

	def resize(self, array, new_size):
		# TODO: add a smoothing element to rezise
		# interpolation types: INTER_NEAREST, INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4
		resized = cv2.resize(array, new_size, interpolation=cv2.INTER_LANCZOS4)
		print("Original:", array.shape, "Resized:", resized.shape)
		fig, ax = plt.subplots(1, 2)  # figsize=[12, 12])
		ax[0].set_title("Original")
		ax[0].imshow(np.array(array), cmap='gray')
		ax[0].axis('off')
		ax[1].set_title("Resized")
		ax[1].imshow(resized, cmap='gray')
		ax[1].axis('off')
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
		return final, mask

	def crop(self, array, mask):
		pad = 4
		row, col = array.shape
		row_min, row_max, col_min, col_max = self.find_edges(mask)
		row_temp = row_max - row_min
		col_temp = col_max - col_min
		temp = row_temp if row_temp >= col_temp else col_temp
		row_temp = row_temp // 2 + row_min
		col_temp = col_temp // 2 + col_min
		row_min = row_temp - temp // 2
		row_max = row_temp + temp // 2
		col_min = col_temp - temp // 2
		col_max = col_temp + temp // 2
		row_min = row_min - pad if row_min - pad >= 0 else 0
		row_max = row_max + pad if row_max + pad <= row - 1 else row - 1
		col_min = col_min - pad if col_min - pad >= 0 else 0
		col_max = col_max + pad if col_max + pad <= col - 1 else col - 1
		array = array[row_min:row_max, col_min:col_max]
		return array

	def find_edges(self, mask_array):
		row, col = mask_array.shape
		row_min = 0
		row_max = row - 1
		col_min = 0
		col_max = col - 1
		for i in range(row):
			if 0 in mask_array[i]:
				row_min = i
				break
		for j in range(row):
			if 0 in mask_array[row - j - 1]:
				row_max = row - j - 1
				break
		c = []
		for i in range(col):
			c.append(i)
		r = []
		for i in range(row):
			r.append(i)
		df = pd.DataFrame(mask_array, columns=c, index=r)
		for i in range(len(df.columns)):
			if 0 in list(df[i]):
				col_min = i
				break
		for j in range(len(df.columns)):
			if 0 in list(df[col - j - 1]):
				col_max = col - j - 1
				break
		return row_min, row_max, col_min, col_max
