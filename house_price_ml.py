import numpy as np

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