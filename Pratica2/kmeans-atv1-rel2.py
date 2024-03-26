import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image

# Load the image
image = Image.open("halteres.jpg")

# Convert the image to a numpy array
image_array = np.array(image)

# Reshape the array to a 2D array of pixels (rows) by RGB values (columns)
reshaped_array = image_array.reshape(-1, 3)

# Define a list of numbers of clusters to try
num_clusters_list = [4,5,6,7]

# Plot the results for each number of clusters
plt.figure(figsize=(15, 5))

for i, num_clusters in enumerate(num_clusters_list):
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(reshaped_array)

    # Get the labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Assign each pixel the color of its cluster's centroid
    clustered_image = centroids[labels].reshape(image_array.shape)

    # Display the clustered image
    plt.subplot(1, len(num_clusters_list), i+1)
    plt.title('Clustered Image ({} clusters)'.format(num_clusters))
    plt.imshow(clustered_image.astype(np.uint8))
    plt.axis('off')

plt.show()