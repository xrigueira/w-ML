import numpy as np
import pandas as pd

"""The code in this file outputs two different files:
    i. anomaly_data.npy: contains the data from each anomaly.
    ii. background_data.npy: contains 5 times the len(anomaly_data)
    of nonanomalous data also know as background."""

# Load the DataFrame from your dataset
station = 901
data = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8', parse_dates=['date'])

# Filter rows where label is 1 (indicating anomalies)
anomalies_df = data[data['label'] == 1]

# Draw inspiration from the anomaly plotter
# Improve https://www.pythoncharts.com/
    
# Filter the data to select only rows where the label column has a value of 1
data_anomalies = data[data["label"] == 1]
    
# Create a new column with the difference between consecutive dates
date_diff = (data_anomalies['date'] - data_anomalies['date'].shift()).fillna(pd.Timedelta(minutes=15))

# Create groups of consecutive dates
date_group = (date_diff != pd.Timedelta(minutes=15)).cumsum()

# Get the starting and ending indexes of each group of consecutive dates
grouped = data.groupby(date_group)
consecutive_dates_indexes = [(group.index[0], group.index[-1]) for _, group in grouped]

# Trim the start and end of the anomalies
trimmed_anomalies_indexes = []
for start, end in consecutive_dates_indexes:
    anomaly_length = end - start
    trim_amount = int(anomaly_length * 0.15)
    trimmed_start = start + trim_amount
    trimmed_end = end - trim_amount
    trimmed_anomalies_indexes.append((trimmed_start, trimmed_end))

# Extract the data
trimmed_data = []

for start, end in trimmed_anomalies_indexes:
    subset_rows = data.iloc[start:end + 1, 1:-2].values.flatten()  # Extract rows within the subset
    trimmed_data.append(subset_rows)

# Save trimmed_data to disk as numpy object
np.save('anomaly_data.npy', trimmed_data, allow_pickle=False, fix_imports=False)
