# Multiple Linear Regression
import csv
import numpy as np
import math

# print(os.listdir(os.curdir))

# Initialize lists to store data
x_train,y_train = [], []

# Open the CSV file and read data
with open('Student_Marks.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip header if exists
    for row in csvreader:
        x_train.append([float(row[0]),float(row[1])])
        y_train.append(float(row[2]))

# Convert lists to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

# Describe the Model
def model(x_train, weights, bias):
    result = np.dot(x_train, weights) + bias
    return result

# initialize Weights and Bias
weights = np.zeros(x_train.shape[1])
bias = 0

def compute_cost(x_train, y_train, weights, bias): 
    m = x_train.shape[0]
    cost = 0
    for i in range(m):                                
        f_wb_i = np.dot(x_train[i], weights) + bias           #(n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y_train[i])**2
    cost = cost / (2 * m)   
    return cost

def compute_gradient(x_train, y_train, weights, bias): 
    m,n = x_train.shape           
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):                             
        err = model(x_train[i], weights, bias) - y_train[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * x_train[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw

iterations = 1000
alpha = 0.001
J_history = []
final_weight = weights
final_bias = bias
for i in range(iterations):

    # Calculate the gradient and update the parameters
    dj_db,dj_dw = compute_gradient(x_train, y_train, final_weight, final_bias)   ##None

    # Update Parameters using w, b, alpha and gradient
    final_weight = final_weight - alpha * dj_dw          
    final_bias = final_bias - alpha * dj_db              
    
    J_history.append(compute_cost(x_train, y_train, final_weight, final_bias))
    # Print cost every at intervals 10 times or as many iterations if < 10
    if i % math.ceil(iterations / 10) == 0:
        print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

print(f"b,w found by gradient descent: {final_bias:0.2f},{final_weight} ")
