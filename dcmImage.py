from pathlib import Path
import pydicom
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import Preprocessing
import os
import pandas as pd


class DicomImage:

	@staticmethod
	def load_dicom(folder_path, filename):
		path = folder_path / filename
		try:
			dicom = pydicom.dcmread(path)
		except Exception as e:
			print("Error: ", e)
			dicom = None
		return dicom

	@staticmethod
	def dicom_to_arr(dicom):
		return dicom.pixel_array

	def show_dicom(self, dicom):
		"""
		Displays DICOM image
		:param dicom: DICOM image
		:return: None
		"""
		arr = self.dicom_to_arr(dicom)
		plt.imshow(arr, cmap='gray')
		plt.show()

	@staticmethod
	def show_array(array):
		"""
		Displays array as image
		:param array: numpy array
		"""
		plt.imshow(array, cmap='gray')
		plt.show()

	@staticmethod
	def scale_to_hu(dicom):
		"""
		Converts DICOM image to numpy array in Hounsfield Units
		:param im: DICOM image
		:return: numpy array
		"""
		arr = dicom.pixel_array
		intercept = int(dicom.RescaleIntercept)
		slope = int(dicom.RescaleSlope)
		if slope != 1:
			raise Exception("Rescale Slope is not 1. May cause type issue")
		arr = slope*arr + intercept
		oldmax = np.max(arr)
		oldmin = np.min(arr)
		arr = arr.astype(np.int16)
		if np.max(arr) != oldmax or np.min(arr) != oldmin:
			raise Exception("Rescaling Numpy array caused data conversion. Possible issue.")
		arr[arr < -1000] = -1000
		arr[arr > 2300] = 2300
		return arr

	@staticmethod
	def histogram(im):
		im = im.astype(np.int16)
		plt.hist(im.flatten(), bins=50, range=(im.min(), im.max()))
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
				temp_im = self.load_dicom(filename)		# Loads the next image
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
	dicom_folder_path = Path.cwd() / "exampleImages_S00"
	di = DicomImage()
	pp = Preprocessing()
	f1 = "ID_b13dbe0d6.dcm"
	im1 = di.load_dicom(dicom_folder_path, f1)
	# pp.channel_split(di.scale_to_hu(im1))
	# di.show_dicom(im1)
	arr1 = di.scale_to_hu(im1)
	final_im1, mask1 = pp.make_mask(arr1, display=True)
	# di.show_array(final_im1)
	# arr1 = pp.crop(final_im1, mask1)
	# di.show_array(arr1)
	# arr1 = pp.resize(arr1, (200, 200))

	# f2 = "ID_000a2d7b0.dcm"
	# im2 = di.load_dicom(dicom_folder_path, f2)
	# pp.channel_split(di.scale_to_hu(im2))
	# di.show_dicom(im2)
	# di.export_patient_ids(folder)
	# arr2 = di.scale_to_hu(im2)
	# final_im2, mask2 = pp.make_mask(arr2, display=True)
	# di.show_array(final_im2)

