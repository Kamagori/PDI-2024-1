import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image

# Load the image
image = Image.open("halteres.jpg")

# Convert the image to a numpy array
image_array = np.array(image)

# Reshape the array to a 2D array of pixels (rows) by RGB values (columns)
X = image_array.reshape(-1, 3)

# Calculate distortion for a range of number of clusters
distortions = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    kmeans.fit(X)
    distortions.append(kmeans.inertia_)

# Plot
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.title('Elbow Method')
plt.show()
