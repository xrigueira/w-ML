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

# Plot good histogram of len anomalies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

data = pd.read_csv('anomalies_length.csv', sep=';', encoding='utf-8')

anomaly_length = list(data.Length_days)

# Calculate the mean value
mean_length = np.mean(anomaly_length)

# Create a histogram
n, bins, patches = plt.hist(anomaly_length, bins=36, facecolor='#3d4899', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7)

n = n.astype('int') # it MUST be integer

# Add a red vertical line at the mean value
plt.axvline(mean_length, color='red', linestyle='dashed', linewidth=2, label='Mean Length', ymax=0.97) # Adjust ymax as needed

# Customize the plot
plt.title('Length of the anomalies')
plt.xlabel('Length in days')
plt.ylabel('Frequency')

# Adding grid lines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adding a legend for better understanding
plt.legend(['Mean length'])

# Show the plot
plt.show()