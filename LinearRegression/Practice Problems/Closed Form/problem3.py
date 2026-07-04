# Linear Regression Practice Problem 3
import numpy as np
import matplotlib.pyplot as plt

X = np.array([1,3,5,7]) # Cars Age in years
Y = np.array([20,15,9,5]) # Car Price in $10000

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

# Predicting car price when the car age is 5 years (X = 5)
NEW_INPUT = 9
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
print(f"Predicted Car Price: {Y_Pred}")

# Visualization
plt.scatter(X,Y,color = 'blue',label = 'Data')
plt.plot(X, (X*m) + b, color = 'purple', label = 'Regression Line')
plt.scatter(NEW_INPUT,Y_Pred,color = 'red', label = 'Prediction')
plt.title("Linear Regression: Car Age vs Car Price")
plt.xlabel("Car Age (Years)")
plt.ylabel("Car Price ($10000)")
plt.legend()
plt.show()