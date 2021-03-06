from pathlib import Path
import pydicom
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import Preprocessing
import os
import pandas as pd


class DicomImage:
	def __init__(self, folder):
		self.home_path = Path.cwd()
		self.dicom_image_folder = folder

	def load_Dicom(self, filename):
		path = self.home_path / self.dicom_image_folder / filename
		try:
			im = pydicom.dcmread(path)
		except Exception as e:
			print("Error: ", e)
			im = None
		return im

	def dicom_to_np_array(self, dicom_image):
		return dicom_image.pixel_array

	def show_Dicom(self, dicom_image):
		"""
		Displays DICOM image
		:param dicom_image: DICOM image
		:return: None
		"""
		arr = dicom_image.pixel_array
		plt.imshow(arr, cmap='gray')
		plt.show()

	def show_array(self, array):
		"""
		Displays array as image
		:param array: numpy array
		"""
		plt.imshow(array, cmap='gray')
		plt.show()

	def scale_to_hu(self, im):
		"""
		Converts DICOM image to numpy array in Hounsfield Units
		:param im: DICOM image
		:return: numpy array
		"""
		arr = im.pixel_array.astype(np.int16)
		b = im.RescaleIntercept
		m = im.RescaleSlope
		arr = m*arr + b
		return arr

	def analyze_image(self, im):
		"""
		prints out each values in data set ************takes a long time to run*********
		:param im:numpy array
		:return: None
		"""
		print(im.size)
		print(im.shape)
		values = []
		for x in np.nditer(im):
			if x not in values:
				values.append(int(x))
		values.sort()
		print(values)

	def histogram(self, im):
		scaled = self.scale_to_hu(im)
		scaled = scaled.astype(np.int16)
		plt.hist(scaled.flatten(), bins=50, range=(scaled.min(), scaled.max()))
		plt.xlabel("Hounstfield Units")
		plt.ylabel("Frequency")
		# plt.xticks([-1000, -105, 0, 350], ["Air", 'Fat', "water", "Bone- Cancellous"], rotation=90)
		plt.show()

	def histogram_w_restriction(self, im):
		# most important section of data
		scaled = self.scale_to_hu(im)
		scaled = scaled.astype(np.int16)
		plt.hist(scaled.flatten(), bins=200, range=(0, 100))
		plt.xlabel("Hounstfield Units")
		plt.ylabel("Frequency")
		plt.xticks([0, 13, 15, 20, 20, 35, 37, 40, 45, 50, 65, 75, 85, 100])
		#plt.xticks([-100, 0, 350], ['Fat (-100)', "water - (0)", "Bone - (350)"])#, rotation=90)
		plt.show()

	def export_patient_ids(self,folder):
		"""
		Exports all patient IDs from each image to a CSV
		:param folder : folder name string
		:return ids : string array
		"""
		ids = []  									# Empty Array to store Patient IDs
		for filename in os.listdir(folder):			# Iterates through each image
			if filename.endswith(".dcm"):
				temp_im = self.load_Dicom(filename)		# Loads the next image
				ids.append(temp_im.PatientID)		# Appends the image's PatientID to ids array
				continue
			else:
				continue
		np.array(ids)								# Convert to Numpy Array
		df = pd.DataFrame(ids)						# Convert to Pandas DataFrame
		df.to_csv("patientIDs.csv", header=False)	# Export DataFrame to CSV file. DO NOT have .csv file open elsewhere
		# No header, indexes present
		return ids


if __name__ == '__main__':
	folder = "exampleImages_S00"
	di = DicomImage(folder)
	pp = Preprocessing(di)
	f1 = "ID_000000e27.dcm"
	im1 = di.load_Dicom(f1)
	# di.show_Dicom(im1)
	final_im1, mask1 = pp.make_mask(im1, display=False)
	# di.show_array(np.array(final_im1))
	arr1 = pp.crop(np.array(final_im1), mask1)
	# di.show_array(arr1)
	arr1 = pp.resize(arr1, (200, 200))

	# f2 = "ID_000a2d7b0.dcm"
	# im2 = di.load_Dicom(f2)
	# di.show_Dicom(im2)
	# di.export_patient_ids(folder)
	# pp.make_mask(im2, display=True)

