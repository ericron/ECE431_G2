from csvloader import CSVFile
from dcmImage import DicomImage
from preprocessing import Preprocessing
import cnn
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt


def preprocess_Dicom(folder_path, filename, types, display=True):
	csv = CSVFile()
	dcm = DicomImage
	pp = Preprocessing()
	data_frame = csv.load_CSV(folder_path, filename + ".csv", types)
	ids_w_label = csv.img_ids_w_labels(data_frame, types)  # dictionary
	ids_list = list(ids_w_label.keys())
	print(ids_list)
	for i in range(len(ids_list)):
		dicom_image = dcm.load_dicom(folder_path, ids_list[i]+".dcm")
		img_array = dcm.scale_to_hu(dicom_image)
		img_array, mask = pp.make_mask(img_array)
		img_array = pp.crop(img_array, mask)
		# may be able to get away with 360, 360
		img_array = pp.resize(img_array, (480, 480))
		img_ch3_array = pp.channel_split(img_array)
		img_array = img_array.astype(np.int16)
		cnn.test_torch_Tensors(img_array)
		cnn.test_torch_Tensors(img_ch3_array)

		if display:
			fig, ax = plt.subplots(2, 2, figsize=[12, 6])
			ax[0, 0].set_title("Full Image")
			ax[0, 0].imshow(img_array, cmap='gray')
			ax[0, 0].axis('off')
			ax[0, 1].set_title("Channeled Image")
			ax[0, 1].imshow(img_ch3_array, cmap='gray')
			ax[0, 1].axis('off')
			ax[1, 0].hist(img_array.flatten(), bins=200, range=(-170, 75))
			ax[1, 0].set_yscale("log")
			ax[1, 0].set_ylim(bottom=10, top=10**4)
			ax[1, 1].hist(img_ch3_array[:,:,0].flatten(), bins=200,
			                range=(img_ch3_array[:,:,0].min(), img_ch3_array[:,:,0].max()))
			ax[1, 1].set_yscale('log')
			ax[1, 1].set_ylim(bottom=10, top=10**4)
			plt.tight_layout()
			plt.show()
	print("Run has ended")




if __name__ == '__main__':
	start_time = time.time()

	a = r"C:\Users\ryanb\Desktop\ECE 431 Project\intraparenchymal"
	dataset1_path = Path('C:/', 'Users', 'ryanb', 'Desktop', 'ECE 431 Project', 'intraparenchymal')
	print(dataset1_path)
	dataset1_filename = 'intraparenchymal'
	types = ["intraparenchymal"]
	preprocess_Dicom(dataset1_path, dataset1_filename, types)

	# dataset2_path = Path('C:/', 'Users', 'ryanb', 'Desktop', 'ECE 431 Project', 'no_hemorrhage_1000')
	# dataset2_filename = 'no_hemorrhage_1000'
	# types = ["any"]
	# preprocess_Dicom(dataset2_path, dataset2_filename, types)

	print("CSVloader Run Time:", time.time() - start_time, "Seconds")
