import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset without specifying column names
dataset = pd.read_csv('dados.csv', header=None)  # Assuming there is no header row

# Separate features (B, G, R) and class labels
features = dataset.iloc[:, :-1]  # Select all columns except the last one
labels = dataset.iloc[:, -1]  # Select the last column

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each class separately
for class_label in labels.unique():
    # Get data points for the current class
    class_data = features[labels == class_label]

    # Extract B, G, R channels
    b = class_data.iloc[:, 0]
    g = class_data.iloc[:, 1]
    r = class_data.iloc[:, 2]

    # Plot the data points for the current class
    ax.scatter(b, g, r, label=f'Class {class_label}')

# Set labels and title
ax.set_xlabel('B')
ax.set_ylabel('G')
ax.set_zlabel('R')
ax.set_title('Pixel Colors in 3D Space')

# Add legend
ax.legend()

# Show plot
plt.show()
