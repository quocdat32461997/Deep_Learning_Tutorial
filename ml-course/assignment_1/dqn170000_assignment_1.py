"""
Author: Dat Quoc Ngo
Date: 02/05/2021
NET-ID: dqn170000
"""

# generating synthetic data

# The true function
def f_true(x):
  y = 6.0 * (np.sin(x + 2) + np.sin(2*x + 4))
  return y

import numpy as np                       # For all our math needs
n = 750                                  # Number of data points
X = np.random.uniform(-7.5, 7.5, n)      # Training examples, in one dimension
e = np.random.normal(0.0, 5.0, n)        # Random Gaussian noise
y = f_true(X) + e                        # True labels with noise

import matplotlib.pyplot as plt          # For all our plotting needs
plt.figure()

"""*************************"""

# Plot the data
plt.scatter(X, y, 12, marker='o')           

# Plot the true function, which is really "unknown"
x_true = np.arange(-7.5, 7.5, 0.05)
y_true = f_true(x_true)
plt.plot(x_true, y_true, marker='None', color='r')

# scikit-learn has many tools and utilities for model selection
from sklearn.model_selection import train_test_split
tst_frac = 0.3  # Fraction of examples to sample for the test set
val_frac = 0.1  # Fraction of examples to sample for the validation set

# First, we use train_test_split to partition (X, y) into training and test sets
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac, random_state=42)

# Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac, random_state=42)

# Plot the three subsets
plt.figure()
plt.scatter(X_trn, y_trn, 12, marker='o', color='orange')
plt.scatter(X_val, y_val, 12, marker='o', color='green')
plt.scatter(X_tst, y_tst, 12, marker='o', color='blue')  

"""*************************"""

# Polynomial Basis Functions

# X float(n, ): univariate data
# d int: degree of polynomial  
def polynomial_transform(X, d):
    # convert data to np.array
    X = np.array(X)
    
    # power univariate data to d-th and transpose
    return np.transpose(np.array([np.power(X, i) for i in range(d + 1)]))

# Phi float(n, d): transformed data
# y   float(n,  ): labels
def train_model(Phi, y):
    Phi_t = np.transpose(Phi) # transpose phi
    # comptue weight
    return np.dot(np.dot(np.linalg.inv(np.dot(Phi_t, Phi)),Phi_t), y)

# Phi float(n, d): transformed data
# y   float(n,  ): labels
# w   float(d,  ): linear regression model
def evaluate_model(Phi, y, w):
    # wT@phi -> converted to phi@w
    # this is due to the piece-wise subtraction
    return np.sum(np.power((y - np.dot(Phi, w)), 2)) / len(y)

w = {}               # Dictionary to store all the trained models
validationErr = {}   # Validation error of the models
testErr = {}         # Test error of all the models

for d in range(3, 25, 3):  # Iterate over polynomial degree
    Phi_trn = polynomial_transform(X_trn, d)                 # Transform training data into d dimensions
    w[d] = train_model(Phi_trn, y_trn)                       # Learn model on training data
    Phi_val = polynomial_transform(X_val, d)                 # Transform validation data into d dimensions
    validationErr[d] = evaluate_model(Phi_val, y_val, w[d])  # Evaluate model on validation data
    
    Phi_tst = polynomial_transform(X_tst, d)           # Transform test data into d dimensions
    testErr[d] = evaluate_model(Phi_tst, y_tst, w[d])  # Evaluate model on test data

# Plot all the models
plt.figure()
plt.plot(validationErr.keys(), validationErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testErr.keys(), testErr.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Polynomial degree', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)
plt.axis([2, 25, 15, 60])

plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for d in range(9, 25, 3):
  X_d = polynomial_transform(x_true, d)
  y_d = X_d @ w[d]
  plt.plot(x_true, y_d, marker='None', linewidth=2)

plt.legend(['true'] + list(range(9, 25, 3)))
plt.axis([-8, 8, -15, 15])

"""*******************************"""

# Radial Basis Functions
# X float(n, ): univariate data
# B float(n, ): basis functions
# gamma float : standard deviation / scaling of radial basis kernel
def radial_basis_transform(X, B, gamma=0.1):
    B = np.array(B)
    # compute radial basis
    return np.exp(-gamma * np.square(np.array([B - x for x in X])))

# Phi float(n, d): transformed data
# y   float(n,  ): labels
# lam float      : regularization parameter
def train_ridge_model(Phi, y, lam):
    # transpose phi
    Phi_t = np.transpose(Phi)
    return np.dot(np.dot(
        np.linalg.inv(np.dot(Phi_t, Phi) + np.dot(lam, np.identity(Phi.shape[-1]))), Phi_t), y)

w = {}               # Dictionary to store all the trained models
validationErr = {}   # Validation error of the models
testErr = {}         # Test error of all the models

Phi_trn = radial_basis_transform(X_trn, X_trn, gamma=0.1)
Phi_val = radial_basis_transform(X_val, X_trn, gamma=0.1)
Phi_tst = radial_basis_transform(X_tst, X_trn, gamma=0.1)
for lam in range(-3, 4):  # lambdas
    w[lam] = train_ridge_model(Phi_trn, y_trn, 10 ** lam)
    validationErr[lam] = evaluate_model(Phi_val, y_val, w[lam])  # Evaluate model on validation data
    
    testErr[lam] = evaluate_model(Phi_tst, y_tst, w[lam])  # Evaluate model on test data

# Plot lambda vs validation error
plt.figure()
plt.plot(validationErr.keys(), validationErr.values(), marker='o', linewidth=3, markersize=12)
plt.plot(testErr.keys(), testErr.values(), marker='s', linewidth=3, markersize=12)
plt.xlabel('Lambda values', fontsize=16)
plt.ylabel('Validation/Test error', fontsize=16)
plt.xticks(list(validationErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Test Error'], fontsize=16)

plt.figure()
plt.plot(x_true, y_true, marker='None', linewidth=5, color='k')

for lam in range(-3, 4):
    X_d = radial_basis_transform(x_true, X_trn, gamma = 0.1)
    y_d = X_d @ w[lam]
    plt.plot(x_true, y_d, marker='None', linewidth=2)

plt.legend(['true'] + list(range(-3, 4)), loc = 'lower right')
plt.axis([-8, 8, -15, 15])
