import pandas as pd

class Batch():
    
    def __init__(self, station, search) -> None:
        self.station = station
        self.search = search
    
    def custom_train_test_split(self, X, y, train_size):
        """This function split the data at the 'train size' length. The test set 
        would be all items which index is smaller then this number, while the test 
        size all those items with an index above.
        ----------
        Argument:
        X (np.array): predictor variables.
        y (np.array): labels or target variable.
        train_size (float): defines the size of the train and test sets.
        
        Return:
        X_train: (np.array): variables train set.
        X_test: (np.array): variables test set.
        y_train: (np.array): target train set.
        y_test: (np.array): target test set."""

        # Define train sets
        X_train = X[:int(train_size*len(X))]
        y_train = y[:int(train_size*len(y))]

        # Define test sets
        X_test = X[int(train_size*len(X)):]
        y_test = y[int(train_size*len(y)):]

        return X_train, X_test, y_train, y_test
    
    def reader(self):
        """This method read the data and splits it in training and testing sets.
        ----------
        Arguments:
        None.
        
        Returns:
        X_train: (np.array): variables train set.
        X_test: (np.array): variables test set.
        y_train: (np.array): target train set.
        y_test: (np.array): target test set."""
        
        # Read the data
        data = pd.read_csv(f'data/labeled_{self.station}_cle.csv', sep=',', encoding='utf-8')

        # Convert variable columns to np.ndarray
        X = data.iloc[:, 1:-1].values
        y = data.iloc[:, -1].values

        # Split the data into test and train sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=0)
        # X_train, X_test, y_train, y_test = self.custom_train_test_split(X, y, train_size=0.8)
        
        return X_train, y_train, X_test, y_test
    
    def logreg(self, X_train, y_train, X_test, y_test):
        """This method implements classification with Logistic Regression.
        ----------
        Arguments:
        X_train: (np.array): variables train set.
        X_test: (np.array): variables test set.
        y_train: (np.array): target train set.
        y_test: (np.array): target test set.
        
        Returns:
        None."""
        
        if self.search == True:
    
            # Define the parameters to iterate over
            param_grid = {'penalty': ['l1', 'l2'], 'fit_intercept': [True, False], 'C': [0.1, 1, 10, 50, 100],
                        'intercept_scaling': [0.1, 1, 5, 10], 'class_weight': ['balanced', None], 'max_iter': [64, 128, 256, 1024]}

            # Call grid search
            from sklearn.model_selection import GridSearchCV
            from sklearn.linear_model import LogisticRegression
            grid_search = GridSearchCV(LogisticRegression(solver='liblinear', random_state=0), param_grid=param_grid, cv=5)

            # Fit grid search to the training data
            grid_search.fit(X_train, y_train)

            # Get best params
            best_params = grid_search.best_params_
            best_model = grid_search.best_estimator_
            print('Best params', best_params, '| Best model', best_model)

            # Make predictions on the testing data
            y_hat = best_model.predict(X_test)

        elif self.search == False:
            
            # Call the model
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(solver='liblinear', max_iter=100, random_state=0)

            # Fit the model to the training data
            model.fit(X_train, y_train)

            # Make predictions on the testing data
            y_hat = model.predict(X_test)

        # Get the accuracy of the model
        from sklearn.metrics import accuracy_score, confusion_matrix
        accuracy = accuracy_score(y_test, y_hat)
        print('Accuracy', accuracy)

        # Get the number of rows labeled as anomalies in y_test
        print('Number of anomalies', len([i for i in y_test if i==1]))

        # Display the confusion matrix
        if self.search == True:
            confusion_matrix = confusion_matrix(y_test, best_model.predict(X_test))
        elif self.search == False:
            confusion_matrix = confusion_matrix(y_test, model.predict(X_test))

        print(confusion_matrix)

    def rf(self, X_train, y_train, X_test, y_test):
        """This method implements classification with Random Forest.
        ----------
        Arguments:
        X_train: (np.array): variables train set.
        X_test: (np.array): variables test set.
        y_train: (np.array): target train set.
        y_test: (np.array): target test set.

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
        print('Number of anomalies', len([i for i in y_test if i==1]))

        # Display the confusion matrix
        if self.search == True:
            confusion_matrix = confusion_matrix(y_test, best_model.predict(X_test))
        elif self.search == False:
            confusion_matrix = confusion_matrix(y_test, model.predict(X_test))

        print(confusion_matrix)
    
    def svm(self, X_train, y_train, X_test, y_test):
        """This method implements classification with SVM.
        This method is quite slow on my machine. Too much data.
        ----------
        Arguments:
        X_train: (np.array): variables train set.
        X_test: (np.array): variables test set.
        y_train: (np.array): target train set.
        y_test: (np.array): target test set.

        Returns:
        None."""

        if self.search == True:
            print('Not developed yet.')
            pass

        elif self.search == False:
            
            # Call the model
            from sklearn import svm
            model = svm.SVC(kernel='rbf')

            # Fit the model to the training data
            model.fit(X_train, y_train)

            # Make predictions on the testing data
            y_hat = model.predict(X_test)

        # Get the accuracy of the model
        from sklearn.metrics import accuracy_score, confusion_matrix
        accuracy = accuracy_score(y_test, y_hat)
        print('Accuracy', accuracy)

        # Get the number of rows labeled as anomalies in y_test
        print('Number of anomalies', len([i for i in y_test if i==1]))

        # Display the confusion matrix
        confusion_matrix = confusion_matrix(y_test, model.predict(X_test))
        print(confusion_matrix)

if __name__ == '__main__':
    
    # Create an instance of the 'batch' class
    batch_model = Batch(station=901, search=False)
    
    # Read and split the data
    X_train, y_train, X_test, y_test = batch_model.reader()
    
    # Implement logistic regression
    batch_model.rf(X_train, y_train, X_test, y_test)