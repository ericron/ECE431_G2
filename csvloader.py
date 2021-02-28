from pathlib import Path
import pandas as pd
import time

# TODO: in combination with DicomImage, pull metadata from images and add to excel files

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
			if i >= 10**3:
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
		limiter = [False, False, False, False, False, True]     # always 'add' any column
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
			index_types[df.columns[i+1]] = []
		for index, row in df.iterrows():
			for i in range(len(df.columns[1:])):
				if row[df.columns[i+1]] == 1:
					index_types[df.columns[i + 1]].append(row['ID'])
		return index_types


if __name__ == '__main__':
	start_time = time.time()
	csv = CSVFile()
	type_list = ['intraparenchymal', 'intraventricular']
	train_data = csv.load_CSV("stage_2_train.csv", type_list, pos_id=True)
	print(train_data.head(4))
	dic_of_IDs = csv.index_types(train_data)
	print(dic_of_IDs)
	print("CSVloader Run Time:", time.time() - start_time, "Seconds")
