import matplotlib.pyplot as plt
import cv2
import numpy as np
import scipy.ndimage
from skimage import morphology
from skimage.transform import resize
from skimage.filters import gaussian
import pandas as pd


class Preprocessing:

	@staticmethod
	def resize(array, new_size, display=False, histogram=False):
		# TODO: add a smoothing element to rezise
		resized = resize(array, new_size, preserve_range=True, anti_aliasing=True)
		resized = resized.astype('int32')
		# blur = gaussian(resized, sigma=1, mode='nearest', preserve_range=True)
		# blur = blur.astype('int32')

		if histogram:
			fig, ax = plt.subplots(1, 2, figsize=[12, 6])
			ax[0].set_title("Original Image")
			ax[0].hist(array.flatten(), bins=200, range=(array.min(), array.max()))
			ax[0].set_yscale("log")
			ax[1].set_title("Resized Image")
			ax[1].hist(resized.flatten(), bins=200, range=(resized.min(), resized.max()))
			ax[1].set_yscale("log")
			plt.tight_layout()
			plt.show()
		if display:
			fig, ax = plt.subplots(1, 2, figsize=[12, 6])
			ax[0].set_title("Original Image")
			ax[0].imshow(array, cmap='gray')
			ax[0].axis('off')
			ax[1].set_title("Resized Image")
			ax[1].imshow(resized, cmap='gray')
			ax[1].axis('off')
			plt.tight_layout()
			plt.show()
		return resized

	@staticmethod
	def make_mask(array, display=False):
		kernel1 = np.ones((7, 7), np.uint8)
		erosion = cv2.erode(array, kernel1, iterations=2)
		kernel2 = np.ones((20, 20), np.uint8)
		dilate = cv2.dilate(erosion, kernel2, iterations=2)
		threshhold = np.where(dilate < -500, 0, 1)
		mask = morphology.area_closing(threshhold, 100000, connectivity=1)
		arr_min = np.amin(array)
		array -= arr_min
		final = array * mask
		final += arr_min
		if display:
			fig, ax = plt.subplots(3, 2, figsize=[12, 12])
			ax[0, 0].set_title("Original")
			ax[0, 0].imshow(array, cmap='gray')
			ax[0, 0].axis('off')
			ax[0, 1].set_title("Erosion")
			ax[0, 1].imshow(erosion, cmap='gray')
			ax[0, 1].axis('off')
			ax[1, 0].set_title("Dilation")
			ax[1, 0].imshow(dilate, cmap='gray')
			ax[1, 0].axis('off')
			ax[1, 1].set_title("Threshold")
			ax[1, 1].imshow(threshhold, cmap='gray')
			ax[1, 1].axis('off')
			ax[2, 0].set_title("Mask")
			ax[2, 0].imshow(mask, cmap='gray')
			ax[2, 0].axis('off')
			ax[2, 1].set_title("Masked Image")
			ax[2, 1].imshow(final, cmap='gray')
			ax[2, 1].axis('off')
			plt.show()
		return final, mask

	def channel_split(self, img, display=False):
		Channel1 = img.copy()
		Channel1[Channel1 < -170] = -175
		Channel1[Channel1 > 75] = -175  # 80
		Channel1 += 175
		Channel1 = Channel1.astype(np.uint8)
		Channel2 = img.copy()
		Channel2[Channel2 < 75] = 70
		Channel2[Channel2 > 300] = 70   # 325
		Channel2 -= 70
		Channel2 = Channel2.astype(np.uint8)
		Channel3 = img.copy()
		Channel3[Channel3 < 300] = 280
		Channel3[Channel3 > 2300] = 2300
		Channel3 = Channel3 - 280
		Channel3 = Channel3 / 8
		Channel3 = Channel3.astype(np.uint8)
		if display:
			self.display_channel_split(img, Channel1, Channel2, Channel3)
		img_3_channels = np.dstack((Channel1, Channel2, Channel3))
		return img_3_channels

	@staticmethod
	def display_channel_split(OrigImg, Channel1, Channel2, Channel3):
		fig, ax = plt.subplots(2, 4, figsize=[12, 12])
		ax[0, 0].set_title("Original Image")
		ax[0, 0].imshow(OrigImg, cmap='gray')
		ax[0, 0].axis('off')
		ax[0, 1].set_title("Channel 1")
		ax[0, 1].imshow(Channel1, cmap='gray')
		ax[0, 1].axis('off')
		ax[0, 2].set_title("Channel 2")
		ax[0, 2].imshow(Channel2, cmap='gray')
		ax[0, 2].axis('off')
		ax[0, 3].set_title("Channel 3")
		ax[0, 3].imshow(Channel3, cmap='gray')
		ax[0, 3].axis('off')
		ax[1, 0].hist(OrigImg.flatten(), bins=250, range=(OrigImg.min(), OrigImg.max()))
		ax[1, 0].set_yscale('log')
		ax[1, 1].hist(Channel1.flatten(), bins=250, range=(Channel1.min(), Channel1.max()))
		ax[1, 1].set_yscale('log')
		ax[1, 2].hist(Channel2.flatten(), bins=225, range=(Channel2.min(), Channel2.max()))
		ax[1, 2].set_yscale('log')
		ax[1, 3].hist(Channel3.flatten(), bins=140, range=(Channel3.min(), Channel3.max()))
		ax[1, 3].set_yscale('log')
		plt.tight_layout()
		plt.show()

	def crop(self, array, mask):
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
		if row_min < 0:
			row_max -= row_min
			row_min = 0
		if row_max > row - 1:
			dif = row_max - (row - 1)
			row_max = row - 1
			row_min -= dif
		if col_min < 0:
			col_max -= col_min
			col_min = 0
		if col_max > col - 1:
			dif = col_max - (row - 1)
			col_max = col - 1
			col_min -= dif
		array = array[row_min:row_max, col_min:col_max]
		return array

	@staticmethod
	def find_edges(mask_array):
		row, col = mask_array.shape
		row_min = 0
		row_max = row - 1
		col_min = 0
		col_max = col - 1
		for i in range(row):
			if 1 in mask_array[i]:
				row_min = i
				break
		for j in range(row):
			if 1 in mask_array[row - j - 1]:
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
			if 1 in list(df[i]):
				col_min = i
				break
		for j in range(len(df.columns)):
			if 1 in list(df[col - j - 1]):
				col_max = col - j - 1
				break
		return row_min, row_max, col_min, col_max
