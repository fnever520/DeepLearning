import numpy as np
import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import join, dirname
import matplotlib.pyplot as plt
# from sklearnext.tools import evaluate_binary_imbalanced_experimetns, read_csv_dir, summarize_binary_datasets

dataset_path = join(dirname(__file__),"HW2_data")
csvfile = join(dataset_path,"breast-cancer.csv")
# result_path = join(dirname(__file__), "result")
# imbalanced_datasets = read_csv_dir(dataset_path)
# imbalanced_datasets_summary.to_csv((join(result_path),"imbalanced_datasets_summary.csv"),index=False)

dataset = pd.read_csv(csvfile)
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,-1].values

# split the datasets into 50-50 training/testing set
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size = 0.5, random_state = 0 )

