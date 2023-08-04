#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 20:59:48 2023

@author: nebiyousamuel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Step 1: Load the dataset and perform linear regression
data = pd.read_csv('NFL2022.csv')
x = data['over/under line']
y = data['total score']
coefficients = np.polyfit(x, y, 1)
b0, b1 = coefficients
fitted_line = b0 + b1 * x

# Step 2: Plot the data with the fitted line
plt.scatter(x, y, label='Data')
plt.plot(x, fitted_line, color='red', label='Fitted Line')
plt.xlabel('Over/Under Line')
plt.ylabel('Total Score')
plt.legend()
plt.show()

# Step 3: Plot a histogram of the total scores and over/under line
plt.hist(y, bins=20, alpha=0.5, label='Total Scores')
plt.hist(x, bins=20, alpha=0.5, label='Over/Under Line')
plt.xlabel('Scores')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Step 4: Plot a histogram of the residuals (total scores - over/under line) and fit a normal curve
residuals = y - fitted_line
plt.hist(residuals, bins=20, alpha=0.5, label='Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

# Fit a normal curve to the residuals
mu, sigma = norm.fit(residuals)
x_axis = np.linspace(min(residuals), max(residuals), 100)
y_fit = norm.pdf(x_axis, mu, sigma) * len(residuals) * np.diff(x_axis)[0]
plt.plot(x_axis, y_fit, 'r--', label='Fitted Normal Curve')
plt.legend()
plt.show()

# Step 5: Calculate the standard deviation of the errors in the totals-over/under line
standard_deviation_errors = np.std(residuals)
print(f"Standard deviation of errors: {standard_deviation_errors}")
