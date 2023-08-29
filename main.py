import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tictoc import tictoc

class Model():
    
    def __init__(self, station, window_size, stride, search) -> None:
        self.station = station
        self.window_size = window_size
        self.stride = stride
        self.search = search
        
        self.X = None
        self.y = None
    
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
        
        # Read the data
        data = pd.read_csv(f'data/labeled_{self.station}_pro.csv', sep=',', encoding='utf-8')
        
        # Calculate the maximum valid start index for a window
        max_start_idx = len(data) - self.window_size
        
        # List to store the flattened window rows and the target labels
        flattened_X = []
        y = []
        
        # Iterate through the original DataFrame with the specified stride
        for i in range(0, max_start_idx + 1, self.stride):
            
            # Calculate the end index for the window
            end_idx = i + self.window_size
            
            # Extract the window of rows
            window_data = data.iloc[i:end_idx, 1:7]
            window_label = data.iloc[i:end_idx, -1]
            
            # Convert the window rows to a 2D array and append to flattened_rows
            flattened_X.append(window_data.values.flatten())
            
            # Store the label for each window
            window_label = 1 if np.mean(window_label) >= 0.5 else 0
            y.append(window_label)
        
        # Convert the flattened data and labels list to a Numpy array
        X = np.array(flattened_X)
        y = np.array(y)
        
        # Store the results
        self.X = X
        self.y = y

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
    def rf(self, X_train, y_train, X_test, y_test):
        """This method implements classification with Random Forest.
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
            param_dist = {'n_estimators': [50, 75, 100, 125, 150, 175], 'max_depth': [1, 2, 3, 4, 5, 10, 15, 20, 50, None],
                        'min_samples_split': [2, 4, 6, 8, 10], 'min_samples_leaf': [1, 2, 3, 4, 5]}
            
            from sklearn.model_selection import RandomizedSearchCV
            from sklearn.ensemble import RandomForestClassifier
            rand_search = RandomizedSearchCV(RandomForestClassifier(random_state=0), param_distributions = param_dist, n_iter=5, cv=5)
            
            rand_search.fit(X_train, y_train)
            
            # Get best params
            best_params = rand_search.best_params_
            best_model = rand_search.best_estimator_
            print('Best params', best_params, '| Best model', best_model)
            
            # Make predictions on the testing data
            y_hat = best_model.predict(X_test)


        elif self.search == False:
            
            # Call the model
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(random_state=0)

            # Fit the model to the training data
            model.fit(X_train, y_train)

            # Make predictions on the testing data
            y_hat = model.predict(X_test)
        
        # Get the accuracy of the model
        from sklearn.metrics import accuracy_score, confusion_matrix
        accuracy = accuracy_score(y_test, y_hat)
        print('Accuracy', accuracy)

        # Get the number of rows labeled as anomalies in y_test
        num_anomalies = len([i for i in y_test if i==1])
        print('Number of anomalies', num_anomalies)

        # Display the confusion matrix
        if self.search == True:
            confusion_matrix = confusion_matrix(y_test, best_model.predict(X_test))
        elif self.search == False:
            confusion_matrix = confusion_matrix(y_test, model.predict(X_test))
            tn, fp, fn, tp = confusion_matrix.ravel()
        
        print(confusion_matrix)
        
        return num_anomalies, tn, fp, fn, tp


if __name__ == '__main__':
    
    # Create an instance of class
    model = Model(station=901, window_size=9, stride=4, search=False)
    
    # Build the windows
    model.windower()
    
    # Shuffle and split the data in train and test sets
    X_train, y_train, X_test, y_test = model.splitter()
    
    num_anomalies, tn, fp, fn, tp = model.rf(X_train, y_train, X_test, y_test)
    
