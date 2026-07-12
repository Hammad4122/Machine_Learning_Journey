# Linear Regression Practice Problem 1
import numpy as np
import matplotlib.pyplot as plt

X = np.array([2,4,6]) # Hours Studied
Y = np.array([50,60,70]) # The Score/Marks

# Mean
X_Mean = np.mean(X)
Y_Mean = np.mean(Y)

# Devitation
X_Dev = X - X_Mean
Y_Dev = Y - Y_Mean

# Product of Deviation
PROD_Dev = X_Dev * Y_Dev

# Sum of product of deviations
total = 0
for i in PROD_Dev:
    total = total + i
PROD_Dev_Sum = total

# Squared Deviation
Squared_Dev = np.sum(X_Dev**2)

# Slope
m = PROD_Dev_Sum/Squared_Dev

# Intercept
b = Y_Mean - (m * X_Mean)

# Predicting scores when the studied time is 5hrs (X = 5)
NEW_INPUT = 5
Y_Pred = (m * NEW_INPUT) + b

# Calculations
print("Calculations:")
print(f"Mean of X: {X_Mean}")
print(f"Mean of Y: {Y_Mean}")
print(f"X Deviation: {X_Dev}")
print(f"Y Deviation: {Y_Dev}")
print(f"Sum of product of deviations: {PROD_Dev_Sum}")
print(f"Sum of squared deviations: {Squared_Dev}")

# Results
print("Results:")
print(f"Slope: {m}")
print(f"Intercept: {b}")
print(f"Predicted Score for 5 hours of study: {Y_Pred}")

# Visualization
plt.scatter(X,Y,color = 'blue',label = 'Data')
plt.plot(X, (X*m) + b, color = 'purple', label = 'Regression Line')
plt.scatter(NEW_INPUT,Y_Pred,color = 'red', label = 'Prediction')
plt.title("Linear Regression: Hours Studied vs Score")
plt.xlabel("Hours Studied")
plt.ylabel("Score/Marks")
plt.legend()
plt.show()