from pathlib import Path
import pandas as pd

# TODO: create data frames limited by type of hemorrage & by list of image ID's to create smaller data sets

# TODO: in combination with DicomImage, pull metadata from images and add to excel files

class CSVFile:
	def __init__(self):
		self.home_path = Path.cwd()
		self.csv_files = []

	def load_CVS(self, filename):
		path = self.home_path / filename
		i = 0
		vals = []
		results = []
		new_col = ['ID', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']
		new_dataframe = pd.DataFrame(columns=new_col)
		for df in pd.read_csv(path, chunksize=6):
			df.reset_index(drop=True, inplace=True)
			vals.clear()
			results.clear()
			name = str(df['ID'][1])
			name = name.split('_')
			name = name[0] + '_' + name[1]
			vals = list(df['Label'])
			results.append(name)
			# TODO: Check what is 1 and 0. Add only selectable types of hemorrhaging
			for a in range(len(vals)):
				results.append(vals[a])
			# TODO: Inefficient - requries making an extra dataframe - not good for large files
			temp_df = pd.DataFrame([results], columns=new_col)
			new_dataframe = new_dataframe.append(temp_df, ignore_index=True)
			i += 1
			if i >= 1000:
				break
		print(new_dataframe)
		self.csv_files = [new_dataframe]


if __name__ == '__main__':
	csv = CSVFile()
	csv.load_CVS("stage_2_train.csv")
