import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

#this function calculates the cost function
def compute_cost(X, y, w, b):
    "X: Features array"
    "y: Target(Actual) values"
    "w: parameter"
    "b: parameter"
    m = X.shape[0]
    total_cost = 0

    for i in range(m):
        f_wb = np.dot(w, X[i]) + b
        y_i = y[i]
        total_cost += (f_wb - y_i)**2

    total_cost /= (2*m)

    return total_cost

# this function computes w and b for gradient descent
def compute_gradient(X, y, w, b):
    "X: Features array"
    "y: Target(Actual) values"
    "w: parameter"
    "b: parameter"

    m = X.shape[0]

    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = np.dot(w, X[i]) + b
        dj_dw += (f_wb - y[i]) * X[i]
        dj_db += (f_wb - y[i])

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db

#this function calculates gradient descent
def gradient_descent(X, y, w_init, b_init, alpha, num_iters):
    m = len(X)

    J_history = []
    w_history = []
    w = np.copy(w_init)
    b = b_init

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)

        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)

        if i<100000:      # prevent resource exhaustion 
            cost =  compute_cost(X, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w, b, J_history, w_history #return w and J,w history for graphing

#this function normalize the features
def zscore_normalize_features(X):

    # Convert to numpy array with explicit dtype
    X = np.array(X, dtype=np.float64)
    
    # find the mean of each column/feature
    mu = np.mean(X, axis=0)
    # find the standard deviation of each column/feature
    sigma = np.std(X, axis=0)
    
    # Avoid division by zero
    sigma = np.where(sigma == 0, 1, sigma)
    
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      

    return (X_norm, mu, sigma)

#this function predicts the value
def predict(X, w, b):
     m = X.shape[0]
     p = np.zeros(m)

     for i in range(m):
          p[i] = np.dot(w, X[i]) + b

     return p

#this function helps in evaluating model(calculating mean square error, root mean squared error, mean absolute error, r sqaure)
def evaluate_model(y_true, y_pred):
      # Mean Squared Error
      mse = np.mean((y_true - y_pred) ** 2)

      # Root Mean Squared Error  
      rmse = np.sqrt(mse)

      # Mean Absolute Error
      mae = np.mean(np.abs(y_true - y_pred))

      # R-squared (coefficient of determination)
      ss_res = np.sum((y_true - y_pred) ** 2)
      ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
      r2 = 1 - (ss_res / ss_tot)

      return mse, rmse, mae, r2

#this function plots the graph
def plot_cost_history(J_history):
      plt.figure(figsize=(10, 6))
      plt.plot(J_history)
      plt.title('Cost vs Iterations')
      plt.xlabel('Iterations')
      plt.ylabel('Cost')
      plt.grid(True)
      plt.show()

#this functions loads training data from data/train.csv
def load_housing_data():
      import os
      # Try different possible paths for the CSV file
      possible_paths = [
          'data/train.csv',
          './data/train.csv',
          os.path.join(os.path.dirname(__file__), 'data', 'train.csv')
      ]
      
      for path in possible_paths:
          try:
              data = pd.read_csv(path)
              print(f"Successfully loaded data from: {path}")
              print(f"Dataset shape: {data.shape}")
              return data
          except FileNotFoundError:
              print(f"File not found at: {path}")
              continue
      
      # If no file found, raise error
      raise FileNotFoundError("train.csv not found. Please ensure the data/train.csv file is available.")


#this function splits data into training and testing sets
def split_data(X, y, test_size=0.2):
    m = X.shape[0]
    indices = np.random.permutation(m)
    split_point = int(m * (1 - test_size))
    
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

#this function preprocesses the training data
def preprocess_data(data):
    selected_features = [
        'GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 
        'OverallQual', 'OverallCond', 'YearBuilt', 'GarageCars',
        'MSZoning', 'Neighborhood', 'KitchenQual', 'CentralAir'
    ]
    
    df = data[selected_features + ['SalePrice']].copy()
    
    # Handle missing values
    numerical_cols = ['GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 
                      'OverallQual', 'OverallCond', 'YearBuilt', 'GarageCars']
    
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    
    categorical_cols = ['MSZoning', 'Neighborhood', 'KitchenQual', 'CentralAir']
    
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Encode categorical variables
    df['CentralAir'] = df['CentralAir'].map({'Y': 1, 'N': 0})
    
    kitchen_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    df['KitchenQual'] = df['KitchenQual'].map(kitchen_mapping)
    
    # One-hot encode MSZoning
    df = pd.get_dummies(df, columns=['MSZoning'], prefix='Zone')
    
    # Neighborhood - keep top 10 most common
    top_neighborhoods = df['Neighborhood'].value_counts().head(10).index
    df['Neighborhood_Popular'] = df['Neighborhood'].apply(
        lambda x: 1 if x in top_neighborhoods else 0
    )
    df.drop('Neighborhood', axis=1, inplace=True)
    
    # Separate features and target
    y = df['SalePrice'].values
    X = df.drop('SalePrice', axis=1).values
    feature_names = df.drop('SalePrice', axis=1).columns.tolist()
    
    print(f"Preprocessed data shape: {X.shape}")
    
    return X, y, feature_names


