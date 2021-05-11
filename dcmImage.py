import numpy as np
import pydicom


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
	def dicom_to_numpy(dicom):
		return dicom.pixel_array

	@staticmethod
	def scale_to_hu(dicom):
		"""
		Converts DICOM image to numpy array in Hounsfield Units
		:param dicom: DICOM image
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
		# there should be no reduction/scaling as a result of conversion to type int16
		arr = arr.astype(np.int16)
		if np.max(arr) != oldmax or np.min(arr) != oldmin:
			raise Exception("Rescaling Numpy array caused data conversion. Possible issue.")
		arr[arr < -1000] = -1000
		arr[arr > 2300] = 2300
		return arr
