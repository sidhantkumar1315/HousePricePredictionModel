import numpy as np
import math

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