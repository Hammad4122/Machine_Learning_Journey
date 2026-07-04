import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

# Data
X = np.array([1, 2, 3, 4, 5, 6])
Y = np.array([35, 45, 50, 60, 65, 80])

m = 6

# Closed form solution
X_Mean = np.mean(X)
Y_Mean = np.mean(Y)
X_Dev = X - X_Mean
Y_Dev = Y - Y_Mean
PROD_Dev_Sum = np.sum(X_Dev * Y_Dev)
Squared_Dev = np.sum(X_Dev**2)
w_closed = PROD_Dev_Sum / Squared_Dev
b_closed = Y_Mean - (w_closed * X_Mean)

# Build the grid of candidate w, b values (the "bowl" background)
w_range = np.linspace(-5, 20, 100)
b_range = np.linspace(-20, 60, 100)
W, B = np.meshgrid(w_range, b_range)

Z = np.zeros_like(W)
for row in range(W.shape[0]):
    for col in range(W.shape[1]):
        w_val = W[row, col]
        b_val = B[row, col]
        Y_pred_grid = (w_val * X) + b_val
        Z[row, col] = (1 / (2 * m)) * np.sum((Y_pred_grid - Y) ** 2)

# Set up 3D figure
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# Draw the bowl surface once (semi-transparent so the descent path is visible through it)
ax.plot_surface(W, B, Z, cmap='viridis', alpha=0.6, edgecolor='none')
ax.set_xlabel('Weight (w)')
ax.set_ylabel('Intercept (b)')
ax.set_zlabel('Cost (MSE)')
ax.set_title('Cost Surface J(w, b) with Gradient Descent Path')

# Gradient descent
ALPHA = 0.05
w, b = 0, 0
iterations = 1000
w_history = [w]
b_history = [b]
cost_history = []

for i in range(iterations):
    Y_Pred = (w * X) + b

    dw = (1 / m) * np.sum((Y_Pred - Y) * X)
    db = (1 / m) * np.sum(Y_Pred - Y)

    w = w - (ALPHA * dw)
    b = b - (ALPHA * db)

    w_history.append(w)
    b_history.append(b)

    cost = (1 / m) * np.sum((Y_Pred - Y) ** 2)
    cost_history.append(cost)

# Cost at each visited (w, b) point along the path, needed to plot the path
# in 3D (x=w, y=b, z=cost at that point)
path_cost = []
for i in range(len(w_history) - 1):
    Y_p = w_history[i] * X + b_history[i]
    path_cost.append((1 / (2 * m)) * np.sum((Y_p - Y) ** 2))
path_cost.append(path_cost[-1])  # repeat last value so lengths match

# Draw the descent path on top of the surface
ax.plot(w_history, b_history, path_cost, color='red', linewidth=2, label='Descent path')
ax.scatter([w], [b], [path_cost[-1]], color='red', s=50, label='Final (w, b)')
ax.legend()

plt.show()

print(f"Closed-form: w={w_closed:.4f}, b={b_closed:.4f}")
print(f"Gradient descent: w={w:.4f}, b={b:.4f}")