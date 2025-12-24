import numpy as np
import math


def initialize_clusters(data, k):
    number_of_dimensions = data.shape[1]
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    centers = np.random.randn(k, number_of_dimensions) * std + mean
    return centers


def assign_clusters(data, cluster_centers):
    distances = np.linalg.norm(data[:, np.newaxis] - cluster_centers, axis=2)
    cluster_table = np.argmin(distances, axis=1)
    return cluster_table


def calculate_cluster_centers(data, assignments, k):
    centers_new = np.zeros((k, data.shape[1]))

    for i in range(k):
        mask = assignments == i
        if np.any(mask):
            centers_new[i] = np.mean(data[mask], axis=0)

    return centers_new


def kmeans(data, initial_cluster_centers):
    k = initial_cluster_centers.shape[0]
    number_of_data = data.shape[0]
    summ = 0

    old_assignments = np.zeros(number_of_data)

    while True:
        assignments = assign_clusters(data, initial_cluster_centers)
        if np.array_equal(assignments, old_assignments):
            break

        initial_cluster_centers = calculate_cluster_centers(data, assignments, k)
        old_assignments = assignments

    for i, datas in enumerate(data):
        summ += np.linalg.norm(datas - initial_cluster_centers[assignments[i]]) ** 2

    return initial_cluster_centers, summ


def visualize(cluster_centers, assignments, data_new, k, orig_shape):
    for i in range(k):
        data_new[assignments == i] = cluster_centers[i]

    data_new = data_new.reshape(orig_shape)
    return data_new


def calculate_error(error):
    error_np = np.array([math.sqrt(i) / 221850 for i in error])
    return error_np
