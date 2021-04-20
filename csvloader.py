import numpy as np
import pandas as pd


class CSVFile:
	def __init__(self):
		self.only_neg = None

	def load_CSV_for_mod(self, filepath, filename, types, pos_id, only_neg):
		"""
		type options: ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
		*empty types list results in dataframe of 'ID'
		:param filepath: path to CSV file
		:param filename: name of CSV file
		:param types: list of types of hemorrhaging to include in the dataframe
		:param pos_id: Boolean; If true, dataframe only includes positivly identified hemorrhages
		:return: dataframe of CSV file
		"""
		path = filepath / filename
		self.only_neg = only_neg
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
			if only_neg:
				if len(results) > 2:
					if 0 in results[1:-1]:
						i += 1
						new_dataframe.loc[len(new_dataframe.index)] = results
				elif len(results) == 2:
					if results[1] == 0:
						i += 1
						new_dataframe.loc[len(new_dataframe.index)] = results
				else:
					i += 1
					new_dataframe.loc[len(new_dataframe.index)] = results
			else:
				if pos_id and len(results) > 2:
					if 1 in results[1:-1]:
						i += 1
						new_dataframe.loc[len(new_dataframe.index)] = results
				elif pos_id and len(results) == 2:
					if results[1] == 1:
						i += 1
						new_dataframe.loc[len(new_dataframe.index)] = results
				else:
					i += 1
					new_dataframe.loc[len(new_dataframe.index)] = results
			# approx: 0.004 x 10^x seconds for 10^x loops
			# aprox 2x that to actually move images
			if i >= 10:
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
		limiter = [False, False, False, False, False, False]
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
		if 'any' in types:
			limiter[5] = True
		if 'ID' in types:
			types.remove('ID')
		new_col = ['ID'] + types
		new_dataframe = pd.DataFrame(columns=new_col)
		return new_dataframe, limiter

	def index_types(self, df):
		"""
		:param df: input dataframe
		:return: dictionary of image IDs for each type of hemorrhaging
		"""
		index_types = {}
		if self.only_neg:
			for i in range(len(df.columns[1:])):
				index_types[df.columns[i + 1]] = []
			for index, row in df.iterrows():
				for i in range(len(df.columns[1:])):
					if row[df.columns[i + 1]] == 0:
						index_types[df.columns[i + 1]].append(row['ID'])
		else:
			for i in range(len(df.columns[1:])):
				index_types[df.columns[i + 1]] = []
			for index, row in df.iterrows():
				for i in range(len(df.columns[1:])):
					if row[df.columns[i + 1]] == 1:
						index_types[df.columns[i + 1]].append(row['ID'])
		return index_types

	def img_ids_w_labels(self, df, hem_type):
		"""
		:param hem_type: type of hemorrhage (can only be one)
		:param df: input dataframe
		:return: dictionary of image IDs for each type of hemorrhaging
		"""
		id_w_label = {}
		for index, row in df.iterrows():
			id_w_label[str(row['ID'])] = int(row[hem_type])
		return id_w_label

	def save_dataframe_as_csv(self, df, filename):
		"""
		Saves dataframe as a CSV file
		:param df: dataframe
		:param filename: Name of CSV file
		:return: None
		"""
		df.to_csv(filename, index=False)
