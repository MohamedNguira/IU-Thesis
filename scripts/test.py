import numpy as np
import matplotlib.pyplot as plt

# Transition matrix (s3 loops to itself)
P = np.array([
    [0, 1, 0],  # s1 → s2
    [0, 0, 1],  # s2 → s3
    [0, 0, 1]   # s3 → s3 (fixed from original code)
])

# Feature matrix (s3 = [1,1], sum of s1 and s2 features)
Phi = np.array([
    [1, 0],  # s1
    [0, 1],  # s2
    [1, 1]   # s3 (changed from [0,2])
])

# TD(0) parameters
alpha = 0.1
gamma = 0.95
w = np.array([1.0, 1.0])  # Initial weights

weight_history = []

# Simulate transitions: s1 → s2 → s3 → s3 → s3...
states = [0, 1, 2, 2, 2, 2, 2]  # Trajectory reflecting absorbing s3

for _ in range(10000):
    for t in range(len(states)-1):
        s = states[t]
        s_next = states[t+1]
        
        # TD(0) update
        td_error = gamma * Phi[s_next] @ w - Phi[s] @ w
        w += alpha * td_error * Phi[s]
        weight_history.append(w.copy())

# Plot divergence
plt.plot(np.array(weight_history)[:,0], label='w1')
plt.plot(np.array(weight_history)[:,1], label='w2')
plt.xlabel('Iterations')
plt.ylabel('Weight Values')
plt.legend()
plt.show()