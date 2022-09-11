import pandas as pd
import numpy as np


data = pd.read_csv("solarpowergeneration.csv")

del data['wind-direction']
del data['wind-speed']
del data['average-wind-speed-(period)']


#seperate 85-15% seperation fro test train 
train_x = np.array(data.iloc[:2482,:6])
train_y = np.array(data.iloc[:2482,6:])

test_x = np.array(data.iloc[2482:,:6])
test_y = np.array(data.iloc[2482:,6:])



