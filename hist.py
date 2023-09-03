# Plot good histogram of len anomalies

import pandas as pd

import matplotlib.pyplot as plt

data = pd.read_csv('anomalies_length.csv', sep=';', encoding='utf-8')

anomaly_length = list(data.Length_days)

# Create a histogram
n, bins, patches = plt.hist(anomaly_length, bins=36, facecolor='#2ab0ff', edgecolor='#e0e0e0', linewidth=0.5, alpha=0.7)

n = n.astype('int') # it MUST be integer

# Good old loop. Choose colormap of your taste
for i in range(len(patches)):
    patches[i].set_facecolor(plt.cm.viridis(n[i] / max(n)))
    
# Customize the plot
plt.title('Length of the anomalies', fontsize=12)
plt.xlabel('Length in days', fontsize=10)
plt.ylabel('Frequency', fontsize=10)

# Show the plot
plt.show()