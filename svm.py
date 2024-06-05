import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tictoc import tictoc

"""The class defined here and the pipeline are indentical to those in main.
The only difference is that the model is trained with a One-Class Support Vector Machine,
instead of a Random Forest Classifier."""

class Model():
    
    def __init__(self, station, window_size, stride, search) -> None:
        self.station = station
        self.window_size = window_size
        self.stride = stride
        self.search = search
        
        self.smoothed_data = None
        self.X_pred = None
        self.y_pred = None
        self.X = None
        self.y = None
    
    # Smooths a column of data using a moving average with specified window size and stride
    def smooth_column(self, column_data, window_size, stride):
        smoothed_values = []
        for i in range(0, len(column_data), stride):
            window_start = max(0, i - window_size // 2)
            window_end = min(len(column_data), i + window_size // 2 + 1)
            window = column_data[window_start:window_end]
            smoothed_values.append(window.mean())
        return smoothed_values
    
    @tictoc
    def preprocessor(self):
        """This function normalizes and smoothes the data.
        ---------
        Arguments:
        self
        
        Returns:
        smoothed_data (Pandas DataFrame): smoothed data."""
        
        # Read the data
        data = pd.read_csv(f'data/labeled_{self.station}_pro.csv', sep=',', encoding='utf-8')
        
        # Normalize the data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        data.iloc[:, 1:-2] = scaler.fit_transform(data.iloc[:, 1:-2])
        
        # Define the variables needed for smoothing
        window_size = 4
        stride = 1
        smoothed_data = data.copy()
        
        # Smoothed the data using parallelization to speed it up
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor() as executor:
            smoothed_columns = executor.map(self.smooth_column, 
                                            [data[col] for col in data.columns[1:-2]],
                                            [window_size] * (len(data.columns[1:-2])),
                                            [stride] * (len(data.columns[1:-2])))
            
            for col, smoothed_values in zip(data.columns[1:-2], smoothed_columns):
                smoothed_data[col] = smoothed_values
        
        self.smoothed_data = smoothed_data
        # smoothed_data.to_csv(f'data/labeled_{self.station}_smo.csv', encoding='utf-8', sep=',', index=False)
    
    @tictoc
    def windower(self):
        """This function reads the database and creates the
        windows.
        ---------
        Arguments:
        self
        
        Returns:
        X (np.array): data.
        y (np.array): labels.
        """
        
        # Calculate the maximum valid start index for a window
        max_start_idx = len(self.smoothed_data) - self.window_size
        
        # List to store the flattened window rows and the target labels
        flattened_X = []
        y = []
        
        # Iterate through the original DataFrame with the specified stride
        for i in range(0, max_start_idx + 1, self.stride):
            
            # Calculate the end index for the window
            end_idx = i + self.window_size
            
            # Extract the window of rows
            window_data = self.smoothed_data.iloc[i:end_idx, 1:7]
            window_label = self.smoothed_data.iloc[i:end_idx, -1]
            
            # Convert the window rows to a 2D array and append to flattened_rows
            flattened_X.append(window_data.values.flatten())
            
            # Store the label for each window
            window_label = 1 if np.mean(window_label) >= 0.5 else 0
            y.append(window_label)
        
        # Convert the flattened data and labels list to a Numpy array
        X = np.array(flattened_X)
        y = np.array(y)
        
        # Stablish a background:anomaly ratio of 12:1
        # Find indices where y is 1 and 0
        indices_anomalies = np.where(y == 1)[0]
        indices_nonanomalies = np.where(y == 0)[0]
        
        # Randomly sample 12 times as many pairs from y=0 to balance the dataset
        num_samples_nonanomalies = len(indices_anomalies) * 12
        selected_indices_nonanomalies = np.random.choice(indices_nonanomalies, num_samples_nonanomalies, replace=False)
        
        # Combine the selected pairs
        selected_indices = np.concatenate((indices_anomalies, selected_indices_nonanomalies))
        X = X[selected_indices]
        y = y[selected_indices]
        
        # Store the results
        self.X = X
        self.y = y

    @tictoc
    def windower_prediction(self):
        """This function reads the database and creates the
        non-overlapping windows for prediction.
        ---------
        Arguments:
        self
        
        Returns:
        X (np.array): data.
        y (np.array): labels.
        """
        
        # Calculate the maximum valid start index for a window
        max_start_idx = len(self.smoothed_data) - self.window_size
        
        # List to store the flattened window rows and the target labels
        flattened_X = []
        y = []
        
        # Iterate through the original DataFrame with the specified window size to create non-overlapping windows
        for i in range(0, max_start_idx + 1, self.window_size):
            
            # Calculate the end index for the window
            end_idx = i + self.window_size
            
            # Extract the window of rows
            window_data = self.smoothed_data.iloc[i:end_idx, 1:7]
            window_label = self.smoothed_data.iloc[i:end_idx, -1]
            
            # Convert the window rows to a 2D array and append to flattened_rows
            flattened_X.append(window_data.values.flatten())
            
            # Store the label for each window
            window_label = 1 if np.mean(window_label) >= 0.5 else 0
            y.append(window_label)
        
        # Convert the flattened data and labels list to a Numpy array
        X = np.array(flattened_X)
        y = np.array(y)
        
        # This version of X and y will be used for prediction
        self.X_pred = X
        self.y_pred = y
        
    @tictoc
    def splitter(self):
        """This function splits the windowed data into the training
        and test sets.
        ----------
        Arguments:
        self.
        
        Retuns:
        X_train (np.array): data training set.
        y_train (np.array) labels training set.
        X_test (np.array): data test set.
        y_test (np.array): labels test set."""
        
        # Combine data and labels into one array to shuffle together
        combined = np.column_stack((self.X, self.y))
        
        # Shuffle the combined array
        np.random.seed(0)
        np.random.shuffle(combined)

        # Split the shuffled array back into data and labels
        shuffled_X, shuffled_y = combined[:, :-1], combined[:, -1]
        
        # Split the shuffled data into the training and testing set
        X_train, y_train = shuffled_X[:int(len(shuffled_X) * 0.75)], shuffled_y[:int(len(shuffled_X) * 0.75)]
        X_test, y_test = shuffled_X[int(len(shuffled_X) * 0.75):], shuffled_y[int(len(shuffled_X) * 0.75):]
        
        return X_train, y_train, X_test, y_test
    
    @tictoc
    def svm(self, X_train, y_train, X_test, y_test):
        """This method implements classification with a One-Class Support Vector Machine.
        ----------
        Arguments:
        X_train (np.array): data training set.
        y_train (np.array) labels training set.
        X_test (np.array): data test set.
        y_test (np.array): labels test set.
        
        Returns:
        None."""
        
        if self.search == True:
            
            # Define the parameters to iterate over
            param_dist = {'kernel': ['sigmoid'], 
                        'gamma': ['scale', 'auto'],
                        'nu': [0.1, 0.2, 0.3, 0.4, 0.5]
                        } 

            from sklearn.model_selection import RandomizedSearchCV
            from sklearn.svm import OneClassSVM
            model = OneClassSVM()

            from sklearn.metrics import make_scorer, f1_score
            
            # Define a custom scorer because the OneClassSVM assigns -1 for outliers and 1 for inliers, which should be 1 and 0 according to my labels
            def custom_f1_score(y_true, y_pred):
                y_pred = np.where(y_pred == 1, 0, 1)  # replace 1 with the negative class label in y_true, and -1 with the positive class label
                return f1_score(y_true, y_pred)
            
            scorer = make_scorer(custom_f1_score)

            rand_search = RandomizedSearchCV(model, param_distributions = param_dist, scoring=scorer, n_iter=10, cv=5)

            rand_search.fit(X_train, y_train)

            # Get best params
            best_params = rand_search.best_params_
            best_model = rand_search.best_estimator_
            print('Best params', best_params, '| Best model', best_model)

            # Make predictions on the testing data
            y_hat = best_model.predict(X_test)

        elif self.search == False:
            
            # Call the model
            from sklearn.svm import OneClassSVM
            model = OneClassSVM(kernel='linear', nu=0.1, gamma='scale')

            # Fit the model to the training data
            model.fit(X_train)
            
            # Save the model to disk
            filename = 'models/svm_model.sav'
            pickle.dump(model, open(filename, 'wb'))

            # Make predictions on the testing data
            y_hat = model.predict(X_test)
            
            # Turn the predictions into 0s and 1s
            y_hat = np.where(y_hat == -1, 1, 0)
        
        # Get the accuracy of the model
        from sklearn.metrics import accuracy_score, confusion_matrix
        accuracy = accuracy_score(y_test, y_hat)
        print('Accuracy', accuracy)

        # Get the number of rows labeled as anomalies in y_test
        num_anomalies = len([i for i in y_test if i==1])
        print('Number of anomalies', num_anomalies)

        # Display the confusion matrix
        if self.search == True:
            y_hat = best_model.predict(X_test)
            y_hat = np.where(y_hat == -1, 1, 0)
            confusion_matrix = confusion_matrix(y_test, y_hat)
        elif self.search == False:
            y_hat = model.predict(X_test)
            y_hat = np.where(y_hat == -1, 1, 0)
            confusion_matrix = confusion_matrix(y_test, y_hat)
            print(confusion_matrix)
            tn, fp, fn, tp = confusion_matrix.ravel()
        
        print(confusion_matrix)
        
        return num_anomalies, tn, fp, fn, tp
    
    @tictoc
    def predictor(self):
        """
        Achieves prediction with the trained and tested model on
        non-overlapping windows.
        """
        
        # Load the model
        filename = 'models/svm_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        
        # Get the number of rows labeled as anomalies in y_test
        num_anomalies = len([i for i in self.y_pred if i==1])
        print('Number of anomalies', num_anomalies)
        
        # Predict on the whole dataset
        y_hat = loaded_model.predict(self.X_pred)
        y_hat = np.where(y_hat == -1, 1, 0)
        # np.save(f'y_svm_{self.station}.npy', y_hat, allow_pickle=False, fix_imports=False)
        from sklearn.metrics import accuracy_score, confusion_matrix
        confusion_matrix = confusion_matrix(self.y_pred, y_hat)
        print(confusion_matrix)


if __name__ == '__main__':
    
    # Create an instance of the class
    model = Model(station=901, window_size=16, stride=1, search=False)
    
    # Preprocess the data (normalizing and smoothing)
    model.preprocessor()
    
    # Build the windows
    model.windower()
    
    # Build the windows for predictions
    model.windower_prediction()
    
    # Shuffle and split the data in train and test sets
    X_train, y_train, X_test, y_test = model.splitter()

    # Train and test the model
    num_anomalies, tn, fp, fn, tp = model.svm(X_train, y_train, X_test, y_test)
    
    # Predict
    model.predictor()
    