from scipy.io import arff
import pandas as pd
import glob
from os.path import join,dirname

# for file in glob.glob("*.arff"):
#     filename = file.split(".")[0]
#     csvfilename = filename + ".csv"

#     data = arff.loadarff(file)
#     df = pd.DataFrame(data[0])
#     df.to_csv(csvfilename)

dataset_path = join((dirname(__file__)), "HW2_data")
data = join(dataset_path,"vote.numeric.arff")

for file in glob.glob(data):
    filename = file.split(".")[0]
    csvfilename = filename + ".csv"

    data = arff.loadarff(file)
    df = pd.DataFrame(data[0])
    df = df.replace({ 
                      b'democrat'   : 1,  \
                      b'republican': 0 
                    })
    print(list(df.shape)[0])
    df.to_csv(csvfilename)


