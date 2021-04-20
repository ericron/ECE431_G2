import numpy as np
from csvloader import CSVFile
from dcmImage import DicomImage
from preprocessing import Preprocessing
from pathlib import Path


def preprocess_dicom_from_masterlist(csv_path, data_path, types, save_path):
	csv = CSVFile()
	dcm = DicomImage
	pp = Preprocessing()
	train_data = csv.load_CSV_for_mod(csv_path, "stage_2_train.csv", types, pos_id=False, only_neg=True)
	ids_w_label = csv.img_ids_w_labels(train_data, types)
	ids_list = list(ids_w_label.keys())
	print(ids_list)
	for i in range(len(ids_list)):
		dicom_image = dcm.load_dicom(data_path, ids_list[i]+".dcm")
		img_array = dcm.scale_to_hu(dicom_image)
		img_array, mask = pp.make_mask(img_array)
		img_array = pp.crop(img_array, mask)
		# may be able to get away with 360, 360
		img_array = pp.resize(img_array, (480, 480))
		img_ch3_array = pp.channel_split(img_array)
		img_array = img_array.astype(np.int16)
		pp.save_numpy_as_png(img_array, ids_list[i]+"_c1", save_path)
		pp.save_numpy_as_png(img_ch3_array, ids_list[i]+"_c3", save_path)
		if i % 500 == 0:
			print(i, "Images have been processed")
	print("Run has ended")


if __name__ == '__main__':
	type_list = ['any']
	csv_location = Path('E:/', 'rsna-intracranial-hemorrhage-detection', 'rsna-intracranial-hemorrhage-detection')
	data_location = Path('E:/', 'rsna-intracranial-hemorrhage-detection', 'rsna-intracranial-hemorrhage-detection', 'stage_2_train')
	save_location = Path('E:/', 'no_hemorrhage_png')
	preprocess_dicom_from_masterlist(csv_location, data_location, type_list, save_location)
