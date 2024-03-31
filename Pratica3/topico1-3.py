import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.utils import shuffle  # Import shuffle function
from kneed import KneeLocator
from PIL import Image


def image_to_dataset(image_path):
    # Open the image
    img = Image.open(image_path)

    # Convert the image to numpy array
    img_array = np.array(img)

    # Flatten the image array to create dataset
    dataset = img_array.reshape(-1, 3)  # Assuming RGB image, reshape to 2D array

    return dataset


def plot_3d_dataset(dataset):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract R, G, B channels
    r = dataset[:, 0]
    g = dataset[:, 1]
    b = dataset[:, 2]

    ax.scatter(r, g, b, c='b', marker='.')
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    plt.show()


def elbow_method(dataset, max_clusters=10):
    distortions = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(dataset)
        distortions.append(kmeans.inertia_)

    # Use KneeLocator to find the "elbow" point
    kn = KneeLocator(range(1, max_clusters + 1), distortions, curve='convex', direction='decreasing')

    return kn.elbow


def cluster_and_show_image(image_path, num_clusters):
    # Load and normalize the image
    img = Image.open(image_path)
    img = np.array(img, dtype=np.float64) / 255

    # Flatten the image array
    w, h, d = original_shape = tuple(img.shape)
    assert d == 3
    image_array = np.reshape(img, (w * h, d))

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    image_array_sample = shuffle(image_array, random_state=42)[:1000]
    kmeans.fit(image_array_sample)
    labels = kmeans.predict(image_array)

    # Recreate the compressed image
    compressed_palette = kmeans.cluster_centers_
    d_image = compressed_palette[labels]
    d_image = np.reshape(d_image, (w, h, d))

    # Plot the original and compressed images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax = axes.ravel()

    ax[0].imshow(img)
    ax[0].set_title('Original Image')

    ax[1].imshow(d_image)
    ax[1].set_title('Compressed Image ({} colors)'.format(num_clusters))

    for a in ax:
        a.axis('off')

    plt.show()


# Step 1: Transform the image into a dataset
image_path = 'halteres.jpg'  # Replace 'your_image.jpg' with the path to your image
dataset = image_to_dataset(image_path)

# Step 2: Plot 3D image of the dataset
plot_3d_dataset(dataset)

# Step 3: Run the Elbow method and proceed with KMeans clustering
best_num_clusters = elbow_method(dataset)
print("Best number of clusters:", best_num_clusters)

# Cluster the image and show the result
cluster_and_show_image(image_path, best_num_clusters)
