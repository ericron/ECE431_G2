from csvloader import CSVFile
from dcmImage import DicomImage
from preprocessing import Preprocessing
from pathlib import Path
import time


if __name__ == '__main__':
	start_time = time.time()
	csv = CSVFile()
	dataset1_path = Path('C:/', 'Users', 'ryanb', 'Desktop', 'ECE 431 Project', 'intrapar_intravent_train_im')
	type_list = ['intraparenchymal', 'intraventricular']
	train_data = csv.load_CSV(dataset1_path, "intrapar_intravent_train.csv", type_list)
	print(train_data.head(4))
	#dic_of_IDs = csv.index_types(train_data)
	#print(dic_of_IDs)
	ids_w_label = csv.img_ids_w_labels(train_data, 'intraparenchymal')  # dictionary
	ids_list = list(ids_w_label.keys())




	print("CSVloader Run Time:", time.time() - start_time, "Seconds")
