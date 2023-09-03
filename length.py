# Program to calculate the length of each anomaly

import pandas as pd

df = pd.read_csv('anomalies_901.csv', sep=';', encoding='utf-8')

df['Start_date'] = pd.to_datetime(df['Start_date'], dayfirst=True)
df['End_date'] = pd.to_datetime(df['End_date'], dayfirst=True)

def get_length(row):
    
    start_date = row['Start_date']
    end_date = row['End_date']
    interval = pd.DateOffset(minutes=15)
    points = pd.date_range(start=start_date, end=end_date, freq=interval)
    
    return len(points)

df['Length'] = df.apply(get_length, axis=1)

df['Length_days'] = df['Length'] / 96

df.to_csv('anomalies_length.csv', sep=';', index=False)