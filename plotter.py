import pandas as pd
import matplotlib.pyplot as plt

"""This file is used to plot the multivariate
data of each day in a data base."""

# Read the data
station = 901
data = pd.read_csv(f'data/labeled_{station}_smo.csv', sep=',', encoding='utf-8', parse_dates=['date'])

# Set the 'date' column as the DataFrame's index
data.set_index('date', inplace=True)

# Drop not needed columns
data = data.drop(data.columns[7:-1], axis=1)

# Normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data.iloc[:, 1:-2] = scaler.fit_transform(data.iloc[:, 1:-2])

# Group the data by day and iterate over each day
for day, day_data in data.groupby(data.index.date):
    
    # Create a new plot for each day
    fig, ax = plt.subplots()

    # # Plot columns 1 to 6 for the current day
    day_data.iloc[:, 0:-2].plot(ax=ax)

    # Customize the plot labels, title, etc.
    ax.set_title(f'Data for {day}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')

    # Save the image
    fig.subplots_adjust(bottom=0.19)
    fig.savefig(f'images/plot_{station}_{day}.png', dpi=300)
    
    # Close the fig for better memory management
    plt.close(fig=fig)
