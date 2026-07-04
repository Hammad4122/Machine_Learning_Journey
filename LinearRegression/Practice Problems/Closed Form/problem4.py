# Linear Regression Practice Problem 4
import numpy as np
import matplotlib.pyplot as plt

X = np.array([5,10,15,20,25]) # Temprature in degree Celsius
Y = np.array([30,25,18,10,5]) # Hot Chocolate Sales

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

# Predicting hot chocolate sales when the temperature is 30°C (X = 30)
NEW_INPUT = 30
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
print(f"Predicted Hot Chocolate Sales: {Y_Pred}")

# Visualization
plt.scatter(X,Y,color = 'blue',label = 'Data')
plt.plot(X, (X*m) + b, color = 'purple', label = 'Regression Line')
plt.scatter(NEW_INPUT,Y_Pred,color = 'red', label = 'Prediction')
plt.title("Linear Regression: Temperature vs Hot Chocolate Sales")
plt.xlabel("Temperature (°C)")
plt.ylabel("Hot Chocolate Sales")
plt.legend()
plt.show()