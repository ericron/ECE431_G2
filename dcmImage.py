from pathlib import Path
import pydicom
import matplotlib.pyplot as plt


class DicomImage:
	def __init__(self, folder):
		self.home_path = Path.cwd()
		self.dicom_image_folder = folder
		self.dicom_images = []

	def load_Dicom(self, filename):
		path = self.home_path / self.dicom_image_folder / filename
		try:
			im = pydicom.dcmread(path)
		except Exception as e:
			print("Error: ", e)
			im = None
		self.dicom_images.append(im)

	def show_Dicom(self):
		for i in range(len(self.dicom_images)):
			if self.dicom_images[i] is not None:
				arr = self.dicom_images[i].pixel_array
				plt.imshow(arr, cmap='gray')
				plt.show()


if __name__ == '__main__':
	folder = "exampleImages_S00"
	DI = DicomImage(folder)
	f1 = "ID_000000e27.dcm"
	DI.load_Dicom(f1)
	f2 = "ID_000a2d7b0.dcm"
	DI.load_Dicom(f2)
	DI.show_Dicom()
