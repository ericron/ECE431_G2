from pathlib import Path
import pandas as pd
import shutil
import time


class CSVFile:
	def __init__(self):
		self.home_path = Path.cwd()

	def load_CSV(self, filename, types, pos_id):
		"""
		type options: ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
		*empty types list results in dataframe of 'ID' and 'add'
		:param filename: name of CSV file
		:param types: list of types of hemorrhaging to include in the dataframe
		:param pos_id: Boolean; If true, dataframe only includes positivly identified hemorrhages
		:return: dataframe of CSV file
		"""
		path = self.home_path / filename
		i = 0
		vals = []
		results = []
		new_dataframe, limiter = self.sort_types(types)
		for df in pd.read_csv(path, chunksize=6):
			df.reset_index(drop=True, inplace=True)
			vals.clear()
			results.clear()
			name = str(df['ID'][1])
			name = name.split('_')
			name = name[0] + '_' + name[1]
			vals = list(df['Label'])
			results.append(name)
			for a in range(len(vals)):
				if limiter[a]:
					results.append(vals[a])
			if pos_id and len(results) > 2:
				if 1 in results[1:-1]:
					new_dataframe.loc[len(new_dataframe.index)] = results
			elif pos_id and len(results) == 2:
				if results[1] == 1:
					new_dataframe.loc[len(new_dataframe.index)] = results
			else:
				new_dataframe.loc[len(new_dataframe.index)] = results
			i += 1
			# approx: 0.004 x 10^x seconds for 10^x loops
			# aprox 2x that to actually move images
			if i >= 10 ** 4:
				break
		pd.set_option('display.max_columns', None)
		return new_dataframe

	def sort_types(self, types=[]):
		"""
		type options: ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
		*empty types list results in dataframe of 'ID' and 'add'
		:param types: list of types of hemorrhaging to include in the dataframe
		:return: empty dataframe with column of labels, and limiter - list of active data
		"""
		limiter = [False, False, False, False, False, True]  # always 'add' any column
		if 'epidural' in types:
			limiter[0] = True
		if 'intraparenchymal' in types:
			limiter[1] = True
		if 'intraventricular' in types:
			limiter[2] = True
		if 'subarachnoid' in types:
			limiter[3] = True
		if 'subdural' in types:
			limiter[4] = True
		if 'ID' in types:
			types.remove('ID')
		if 'any' in types:
			types.remove('any')
		new_col = ['ID'] + types + ['any']
		new_dataframe = pd.DataFrame(columns=new_col)
		return new_dataframe, limiter

	def index_types(self, df):
		"""
		:param df: input dataframe
		:return: dictionary of image IDs for each type of hemorrhaging
		"""
		index_types = {}
		for i in range(len(df.columns[1:])):
			index_types[df.columns[i + 1]] = []
		for index, row in df.iterrows():
			for i in range(len(df.columns[1:])):
				if row[df.columns[i + 1]] == 1:
					index_types[df.columns[i + 1]].append(row['ID'])
		return index_types

	def save_dataframe_as_csv(self, df, filename):
		"""
		Saves dataframe as a CSV file
		:param df: dataframe
		:param filename: Name of CSV file
		:return: None
		"""
		df.to_csv(filename, index=False)

	def index_types_to_dataset(self, dicom_location, new_dataset_location, dict, subdirectory=False):
		keys = list(dict.keys())
		for i in range(len(keys)):
			hem_type = str(keys[i])
			list_ids = dict[keys[i]]
			if subdirectory:
				new_data_loc = new_dataset_location / hem_type
				self.create_dataset(dicom_location, new_data_loc, list_ids)
			else:
				if not hem_type == 'any':
					self.create_dataset(dicom_location, new_dataset_location, list_ids)

	def create_dataset(self, dicom_location, new_dataset_location, IDs):
		"""
		Copys dicom image from dicom_location to new_dataset_location of dicom images contained
		in list IDs
		:param dicom_location: pathlib Path of current dicom image location
		:param new_dataset_location: pathlib Path of location to copy dicom images to
		:param IDs: list of dicom image IDs (no .dcm on end). format example: ['ID_000af28ac',]
		:return: None
		"""
		if dicom_location.is_dir():
			new_dataset_location.mkdir(exist_ok=True)
			for f in IDs:
				filename = f + '.dcm'
				cur_loc = dicom_location / filename
				if cur_loc.is_file():
					future_loc = new_dataset_location / filename
					if not future_loc.exists():
						shutil.copy(cur_loc, future_loc)
					else:
						print("Error: Dicom image file already exists")
				else:
					print("Error: Dicom image file could not be found")
		else:
			print("Error: Dicom image location folder does not exist")


if __name__ == '__main__':
	start_time = time.time()
	csv = CSVFile()
	type_list = ['intraparenchymal', 'intraventricular']
	train_data = csv.load_CSV("stage_2_train.csv", type_list, pos_id=True)
	print(train_data.head(4))
	dic_of_IDs = csv.index_types(train_data)
	csv_filename = 'intrapar_intravent_train.csv'
	csv.save_dataframe_as_csv(train_data, csv_filename)

	# test_dic = {'intraparenchymal': ["ID_000a2d7b0", "ID_000a8710b", "ID_000a50137", "ID_000bf8860", "ID_000000e27"],
	#             'intraventricular': ["ID_000af28ac", "ID_000b220f4", "ID_000bda502"],
	#             'any': ["ID_000a2d7b0", "ID_000a8710b", "ID_000a50137", "ID_000af28ac", "ID_000b220f4", "ID_000bda502",
	#              "ID_000bf8860", "ID_000000e27"]}
	# test_dic for folder "exampleImages_S00"
	# dic_loc = Path.cwd() / "exampleImages_S00"
	# E:\rsna-intracranial-hemorrhage-detection\rsna-intracranial-hemorrhage-detection\stage_2_train
	dic_loc = Path('E:/', 'rsna-intracranial-hemorrhage-detection', 'rsna-intracranial-hemorrhage-detection',
	               'stage_2_train')
	# new_data_loc = Path.cwd() / "intrapar_intravent_train_im"
	new_data_loc = Path('C:/', 'Users', 'ryanb', 'Desktop', 'ECE 431 Project', 'intrapar_intravent_train_im')
	csv.index_types_to_dataset(dic_loc, new_data_loc, dic_of_IDs)
	print("CSVloader Run Time:", time.time() - start_time, "Seconds")
