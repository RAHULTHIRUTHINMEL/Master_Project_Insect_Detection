from dv import AedatFile
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from sklearn.cluster import DBSCAN


with AedatFile("fly1.aedat4") as f:
    # list all the names of streams in the file
    #print(f.names)

    events = np.hstack([packet for packet in f['events'].numpy()])
    
    #print(events)
    # # Access information of all events by type
    timestamps, x, y, polarities = events['timestamp'], events['x'], events['y'], events['polarity']
     # numberevents = 100
    # for i in range(numberevents):
    #     print(timestamps[i], x[i], y[i], polarities[i])

    # points = [[p[1],p[2],p[3]] for p in events]
    # # print(points)
    # # import sys
    # # sys.exit(0)
    

num_neighbors = 15  # Number of neighbors to consider

x_array = np.array(x)
y_array = np.array(y)


# Assuming you have a NumPy array for x-coordinates (x_array) and y-coordinates (y_array)

coordinates = np.column_stack((x_array[:100], y_array[:100]))  # Combine x and y coordinates

kdtree = cKDTree(coordinates)  # Build the KD-tree

distances, indices = kdtree.query(coordinates, k=15)  # Query the KD-tree for the 15 nearest neighbors (including the point itself)

average_distances = np.mean(distances[:, :])  # Calculate the average distance excluding the point itself

print(average_distances)

std_distance = np.std(average_distances)
threshold = 0.5  # Threshold as a multiple of the standard deviation

outlier_indices = np.where(average_distances > (np.mean(average_distances) + threshold * std_distance))[0]  # Find the outlier indices

filtered_coordinates = coordinates[~np.isin(np.arange(len(coordinates)), outlier_indices)]  # Filter out the outlier coordinates

# Print the filtered coordinates
print(filtered_coordinates)

filtered_x = filtered_coordinates[:, 0]
#print(filtered_x)


filtered_y = filtered_coordinates[:, 1]
print(filtered_y)





# Assuming you have 'filtered_x' and 'filtered_y' as the filtered x and y coordinates

# Combine filtered x and y coordinates into a single array
filtered_coordinates = np.column_stack((filtered_x, filtered_y))

# DBSCAN clustering parameters
eps = 30  # Neighborhood search radius
min_samples = 10  # Minimum number of neighbors required to identify a core point

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
cluster_labels = dbscan.fit_predict(filtered_coordinates)

# Get the number of clusters found
num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)  # Minus 1 for noise cluster

# Print the number of clusters found
print("Number of clusters:", num_clusters)

# Print the cluster labels for each event point
print("Cluster labels:")
print(cluster_labels)





# p = prefilter_events(events)
# print(p)