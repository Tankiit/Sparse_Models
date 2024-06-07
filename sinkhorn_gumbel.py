import numpy as np
import matplotlib.pyplot as plt

# Generate a random 5x5 matrix
W = np.random.rand(5, 5)

# Generate a sparse version of W by setting values below a threshold to zero
threshold = 0.5
W_sparse = W.copy()
W_sparse[W_sparse < threshold] = 0

# Print W_sparse
print("W_sparse:")
print(W_sparse)

# Show the heatmap of W_sparse
fig, ax = plt.subplots(figsize=(6, 6))
plt.imshow(W_sparse, cmap='coolwarm', interpolation='nearest')
plt.title('')
plt.xticks([])  # Remove x-axis ticks
plt.yticks([])  # Remove y-axis ticks
plt.savefig('S_sparse.png', transparent=True)
plt.show()