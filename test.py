import numpy as np

### NORMALIZING ###
# Normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data.iloc[:, 1:-1] = scaler.fit_transform(data.iloc[:, 1:-1])