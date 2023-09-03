import random
import pickle
import numpy as np
import pandas as pd


"""The code in this file outputs two different files:
    i. anomaly_data.npy: contains the data from each anomaly.
    ii. background_data.npy: contains 5 times the len(anomaly_data)
    of nonanomalous data also know as background."""

def anomalies():

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
    anomaly_data = []
    for start, end in trimmed_anomalies_indexes:
        subset_rows = data.iloc[start:end + 1, 1:-2].values.flatten()  # Extract rows within the subset
        anomaly_data.append(subset_rows)
    
    # Save anomalys_data to disk as numpy object
    # Save the list using pickle
    with open('anomaly_data.pkl', 'wb') as file:
        pickle.dump(anomaly_data, file)

def background():
    
    # Load the DataFrame from your dataset
    station = 901
    data = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8', parse_dates=['date'])

    # Filter rows where label is 1 (indicating anomalies)
    len_anomalies = len(data[data['label'] == 1])
    
    # Define the number of windows (days, in this case) to extract
    ratio = 5
    len_window = 96
    num_windows = int((len_anomalies * ratio) / len_window)
    
    # Filter the data to select only rows where the label column has a value of 0
    data_background = data[data["label"] == 0]
    
    mean_ammonium = np.mean(data_background.ammonium_901)
    
    # Filter the dataset to include only days that meet the condition
    filtered_df = data_background.groupby(data_background['date'].dt.date).filter(lambda x: x['ammonium_901'].max() <= mean_ammonium)

    # Get a list of unique days
    unique_days = filtered_df['date'].dt.date.unique()
    
    # Randomly select 173 unique days
    selected_days = random.sample(unique_days.tolist(), num_windows)
    
    background_data = []
    for day in selected_days:
        day_data = filtered_df[filtered_df['date'].dt.date == day]
        day_data = day_data.iloc[:, 1:-2].values.flatten()
        background_data.append(day_data)
    
    # Save anomalies_data to disk as numpy object

    with open('background_data.pkl', 'wb') as file:
        pickle.dump(background_data, file)
        
if __name__ == '__main__':
    
    anomalies()
    
    background()
