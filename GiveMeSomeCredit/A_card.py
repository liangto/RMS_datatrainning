


#%%

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

Data_train=pd.read_csv("train.csv")
# print(Data_train.head())

#%%
print(Data_train.describe())