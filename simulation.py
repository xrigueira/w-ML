import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main import Model

if __name__ == '__main__':
    
    window_sizes = [4]
    strides = [1, 2]
    
    results = pd.DataFrame(columns=['window_size', 'stride', 'anomalies', 'tn', 'fp', 'fn', 'tp'])
    
    for window_size in window_sizes:
    
        for stride in strides:

            # Create an instance of class
            model = Model(station=901, window_size=window_size, stride=stride, search=False)
    
            # Build the windows
            model.windower()
    
            # Shuffle and split the data in train and test sets
            X_train, y_train, X_test, y_test = model.splitter()
            
            num_anomalies, tn, fp, fn, tp = model.rf(X_train, y_train, X_test, y_test)    
            
            results.loc[len(results.index)] = [window_size, stride, num_anomalies, tn, fp, fn, tp]
        
# Save the results
results.to_csv(f'results.csv', sep=',', encoding='utf-8', index=True)