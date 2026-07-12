import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.array([1,2,3,4,5,6])
Y = np.array([35,45,50,60,65,80])

m = 6
# Finding w and b using closed form formula
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
w = PROD_Dev_Sum/Squared_Dev

# Intercept
b = Y_Mean - (w * X_Mean)

# print(w)
# print(b)

# Build the grid of candidate w, b values (the "bowl" background)
w_range = np.linspace(-5,20,100)
b_range = np.linspace(-20,60,100)

W, B = np.meshgrid(w_range, b_range)

Z = np.zeros_like(W)

for row in range(W.shape[0]):
    for col in range(W.shape[1]):
        w_val = W[row, col]
        b_val = B[row, col]
        Y_pred_grid = (w_val * X) + b_val
        Z[row,col] = (1/(2*m)) * (np.sum((Y_pred_grid-Y)**2))


contours = plt.contour(W,B,Z,levels = 60, cmap = 'viridis')
plt.clabel(contours, inline=True, fontsize=8)
plt.xlabel('Weight (w)')
plt.ylabel('Intercept (b)')
plt.title('Cost (MSE) Contours')
plt.pause(0.1)
plt.show()



# Applying Gradient Descent to reduce cost
ALPHA = 0.05; w = 0; b = 0; iterations = 1000; cost_history = []; w_history = [w]; b_history = [b]
for i in range(iterations):
    Y_Pred = (w * X) + b

    w = w - (ALPHA * ((1/m) * np.sum((Y_Pred-Y)*X)))
    b = b - (ALPHA * ((1/m) * np.sum(Y_Pred-Y)))

    w_history.append(w)
    b_history.append(b)

    
    cost = (1/m) * (np.sum((Y_Pred-Y)**2))
    cost_history.append(cost)

    plt.ion()
    if i % 5 == 0 or i == iterations - 1:
        plt.clf()
        contours = plt.contour(W,B,Z,levels = 100, cmap = 'viridis')
        plt.plot(w_history,b_history, color = 'red')
        plt.scatter([w], [b], color = 'red', label = 'Current (m,b)')
        plt.clabel(contours, inline=True, fontsize=8)
        # plt.set_title("Cost surface J(m, b)")
        # plt.set_xlabel("m")
        # plt.set_ylabel("b")
        plt.legend()
        plt.pause(0.01)

    print(f"{w:.2f}, {b:.2f}, {i}")



print(f"{w}, {b}")
plt.ioff()
plt.show()