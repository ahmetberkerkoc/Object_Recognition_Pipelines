import numpy as np
import matplotlib.pyplot as plt


def calculate_distances(train_data, test_datum):
    i = 0
    distance_table = np.zeros(len(train_data))
    for data in train_data:
        distance = np.linalg.norm(test_datum.hist - data.hist)
        distance_table[i] = distance
        i = i + 1
    return distance_table


def majority_voting(distances, labels, k):

    list_of_labels = []
    index1 = np.argpartition(distances, k)
    index = index1[:k]
    for i in range(len(index)):
        list_of_labels.append(labels[index[i]].category)
    unique, position = np.unique(list_of_labels, return_inverse=True)
    max_element_pos = np.bincount(position).argmax()
    majority_class = unique[max_element_pos]
    return majority_class


def knn(train_data, train_labels, test_data, test_labels, k):
    i = 0
    accuracy = 0
    for test_datum in test_data:
        distances = calculate_distances(train_data, test_datum)
        majority_class = majority_voting(distances, train_labels, k)
        if majority_class == test_labels[i]:
            accuracy = accuracy + 1
        i = i + 1
    accuracy = accuracy / (test_data.shape[0])
    return accuracy
