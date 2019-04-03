from scipy.io import arff
import pandas as pd
import glob

for file in glob.glob("*.arff"):
	filename = file.split(".")[0]
	csvfilename = filename + ".csv"

	data = arff.loadarff(file)
	df = pd.DataFrame(data[0])
	df.to_csv(csvfilename)
