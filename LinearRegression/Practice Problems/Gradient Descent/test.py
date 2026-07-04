import numpy as np
import matplotlib.pyplot as plt

# Data (Problem 1)
X = np.array([2, 4, 6])
Y = np.array([50, 60, 70])
n = len(X)

def cost_fn(m, b):
    Y_pred = m * X + b
    return (1/(2*n)) * np.sum((Y_pred - Y)**2)

# Init
m, b = 0, 0
alpha = 0.05
iterations = 4000
cost_history = []
m_history = [m]
b_history = [b]

# Precompute the cost surface (the "bowl") over a grid of possible m, b values
m_range = np.linspace(-5, 15, 100)
b_range = np.linspace(-10, 60, 100)
M, B = np.meshgrid(m_range, b_range)
Z = np.zeros_like(M)
for row in range(M.shape[0]):
    for col in range(M.shape[1]):
        Z[row, col] = cost_fn(M[row, col], B[row, col])

# Set up three side-by-side live plots
plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

for i in range(iterations):
    Y_pred = m * X + b
    error = Y_pred - Y

    # Gradients
    dm = (1/n) * np.sum(error * X)
    db = (1/n) * np.sum(error)

    # Update
    m -= alpha * dm
    b -= alpha * db
    m_history.append(m)
    b_history.append(b)

    # Cost
    cost = (1/(2*n)) * np.sum(error**2)
    cost_history.append(cost)

    # Redraw every 5 iterations (redrawing every single one is too slow)
    if i % 5 == 0 or i == iterations - 1:
        ax1.clear()
        ax1.scatter(X, Y, color='blue', label='Data')
        ax1.plot(X, m*X + b, color='red', label='Fit line')
        ax1.set_title(f"Iteration {i} | m={m:.2f}, b={b:.2f}")
        ax1.set_xlabel("Hours Studied")
        ax1.set_ylabel("Score")
        ax1.legend()
        ax1.set_xlim(0, 8)
        ax1.set_ylim(0, 90)

        ax2.clear()
        ax2.plot(cost_history, color='purple')
        ax2.set_title(f"Cost = {cost:.3f}")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Cost (MSE)")

        # The convex "bowl" -- contour of cost over all (m, b), with the
        # path gradient descent has taken so far traced on top of it
        ax3.clear()
        ax3.contour(M, B, Z, levels=30, cmap='viridis')
        ax3.plot(m_history, b_history, color='blue', linewidth=1.5)
        ax3.scatter([m], [b], color='red', zorder=5, label='Current (m, b)')
        ax3.set_title("Cost surface J(m, b)")
        ax3.set_xlabel("m")
        ax3.set_ylabel("b")
        ax3.legend()
        plt.pause(0.01)

plt.ioff()
plt.show()

print(f"Final slope (m): {m:.4f}")
print(f"Final intercept (b): {b:.4f}")