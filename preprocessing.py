import numpy as np
import cv2
from skimage import morphology, measure
from skimage.transform import resize
import pandas as pd
from PIL import Image


class Preprocessing:

	@staticmethod
	def save_numpy_as_png(np_arr, filename, save_location):
		if len(np_arr.shape) == 2:
			arr_min = np.amin(np_arr)
			np_arr -= arr_min
		pngName = filename + '.png'
		pngPath = save_location / pngName
		img = Image.fromarray(np_arr)
		img.save(pngPath)

	@staticmethod
	def resize(array, new_size):
		resized = resize(array, new_size, preserve_range=True, anti_aliasing=True)
		resized = resized.astype('int32')
		return resized

	@staticmethod
	def make_mask(array):
		kernel1 = np.ones((10, 10), np.uint8)
		erosion = cv2.erode(array, kernel1, iterations=2)
		kernel2 = np.ones((20, 20), np.uint8)
		dilate = cv2.dilate(erosion, kernel2, iterations=2)
		threshhold = np.where(dilate < -500, 0, 1)
		mask = morphology.area_closing(threshhold, 100000, connectivity=1)
		mask_labels = measure.label(mask)
		regions = measure.regionprops(mask_labels)
		if len(regions) > 1:
			regions.sort(key=lambda x: x.area, reverse=True)
			for region in regions[1:]:
				mask[region.coords[:, 0], region.coords[:, 1]] = 0
		arr_min = np.amin(array)
		array -= arr_min
		final = array * mask
		final += arr_min
		return final, mask

	def channel_split(self, img):
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
		img_3_channels = np.dstack((Channel1, Channel2, Channel3))
		return img_3_channels

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
