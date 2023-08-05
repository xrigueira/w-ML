import numpy as np
import pandas as pd

window_sizes = [16, 24, 32, 64, 96]
strides = [1, 2, 4, 8]

results = pd.DataFrame(columns=['window_size', 'stride', 'anomalies', 'tn', 'fp', 'fn', 'tp'])
for window_size in window_sizes:
    
    for stride in strides:
        
        anomalies = np.random.uniform(0, 200, 1)
        tn = np.random.uniform(0, 1, 1)
        fp = np.random.uniform(0, 1, 1)
        fn = np.random.uniform(0, 1, 1)
        tp = np.random.uniform(0, 1, 1)
        
        results.loc[len(results.index)] = [window_size, stride, anomalies, tn, fp, fn, tp]
        
# Save the results
results.to_csv(f'results.csv', sep=',', encoding='utf-8', index=True)