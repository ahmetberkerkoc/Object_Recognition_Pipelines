import numpy as np
from copy import deepcopy


NUM_CLASS = 3
COUNTER = 0
MAX_DEPTH = 8
CRITERION = "entropy"
PRUNING = True
PRUNUNG_THRESHOLD = 10


class Node:
    def __init__(self, split_point, labels, feature, depth):
        self.split_point = split_point
        self.labels = labels
        self.left = None
        self.right = None
        self.feature = feature
        self.depth = depth
        self.n = np.sum(labels)

    def PrintTree(self, column_names_dic):
        if self.left:
            self.left.PrintTree()

        added_string = " " * self.depth * 4
        if self.left is not None or self.right is not None:
            print(
                added_string
                + "split value: {}, bucket: [{}, {}, {}], attribute: {} ".format(
                    self.split_point, self.labels[0], self.labels[1], self.labels[2], column_names_dic[self.feature]
                )
            )
        else:
            print(
                added_string
                + "LEAF! bucket: [{}, {}, {}], attribute: {} ".format(
                    self.labels[0], self.labels[1], self.labels[2], column_names_dic[self.feature]
                )
            )
        if self.right:
            self.right.PrintTree()


def entropy(bucket):
    total = sum(bucket)
    if total == 0:
        return 0
    E = 0
    for i in range(len(bucket)):
        p = bucket[i] / total
        if p != 0:
            E += -p * np.log2(p)
    return E


def entropy_info_gain(parent_bucket, left_bucket, right_bucket):

    parent_ent = entropy(parent_bucket)
    left_ent = entropy(left_bucket)
    right_ent = entropy(right_bucket)
    example_size = np.sum(parent_bucket)
    gain = (
        parent_ent - (np.sum(left_bucket) / example_size) * left_ent - (np.sum(right_bucket) / example_size) * right_ent
    )
    return gain


def gini(bucket):

    total = sum(map(lambda i: i * i, bucket)) / (sum(bucket) ** 2)
    gini_index = 1 - total

    return gini_index


def avg_gini_index(left_bucket, right_bucket):

    left_gini = gini(left_bucket)
    right_gini = gini(right_bucket)
    size_left = sum(left_bucket)
    size_right = sum(right_bucket)
    total_size = size_left + size_right

    avg = left_gini * (size_left / total_size) + right_gini * (size_right / total_size)
    return avg


def calculate_labels(labels, num_class):

    bucket = [0] * num_class
    for l in labels:
        bucket[l - 1] += 1
    return bucket


def find_best_split_value(train_data, train_labels, num_class, feature_index, criterion):

    feature_values = deepcopy(train_data[:, feature_index])
    feature_values = np.sort(feature_values)
    feature_values = np.unique(feature_values)

    p_node = calculate_labels(train_labels, num_class)
    p_gini = gini(p_node)

    best_criterion_gain = -float("inf")
    best_split_point = -float("inf")
    for i in range(len(feature_values) - 1):
        data_right, label_right, data_left, label_left = [], [], [], []
        split_point = (feature_values[i] + feature_values[i + 1]) / 2
        for data, label in zip(train_data, train_labels):
            if data[feature_index] >= split_point:
                data_right.append(data)
                label_right.append(label)
            else:
                data_left.append(data)
                label_left.append(label)

        l_node = calculate_labels(label_left, num_class)
        r_node = calculate_labels(label_right, num_class)
        if criterion == "gini":
            new_gini_value = avg_gini_index(l_node, r_node)
            gini_gain = p_gini - new_gini_value

            if gini_gain > best_criterion_gain:
                best_criterion_gain = gini_gain
                best_split_point = split_point

        elif criterion == "entropy":
            entropy_gain = entropy_info_gain(p_node, l_node, r_node)

            if entropy_gain > best_criterion_gain:
                best_criterion_gain = entropy_gain
                best_split_point = split_point

    return best_criterion_gain, best_split_point


def find_best_split(train_data, train_labels, num_class, features, criterion):

    gain_list = []
    split_list = []
    for feature in features:
        gini_gain, split_point = find_best_split_value(train_data, train_labels, num_class, feature, criterion)
        gain_list.append(gini_gain)
        split_list.append(split_point)
    max_gain = max(gain_list)
    max_index = gain_list.index(max_gain)
    best_feature = features[max_index]
    best_split_point = split_list[max_index]
    return (best_feature, best_split_point)


def split_data(train_data, train_labels, feature, split_point):
    d_right, d_left, l_right, l_left = [], [], [], []
    for data, label in zip(train_data, train_labels):
        if data[feature] >= split_point:
            d_right.append(data)
            l_right.append(label)
        else:
            d_left.append(data)
            l_left.append(label)
    return (np.array(d_right), np.array(l_right)), (np.array(d_left), np.array(l_left))


def make_tree(train_data, train_labels, num_class, features, depth, criterion):
    global COUNTER
    COUNTER = COUNTER + 1
    root = None
    labels = []
    for n in range(num_class):
        labels.append(np.count_nonzero(train_labels == n + 1))

    best_split_feature, best_split_value = find_best_split(train_data, train_labels, num_class, features, criterion)

    root = Node(best_split_value, labels, best_split_feature, depth)

    if PRUNING and root.n < PRUNUNG_THRESHOLD:
        return root

    if labels.count(0) >= 2:
        return root
    else:
        right_side, left_side = split_data(train_data, train_labels, best_split_feature, best_split_value)
        left_labels = calculate_labels(left_side[1], num_class)
        right_labels = calculate_labels(right_side[1], num_class)

        if depth + 1 <= MAX_DEPTH:

            root.left = make_tree(left_side[0], left_side[1], num_class, features, depth + 1, criterion)
            root.right = make_tree(right_side[0], right_side[1], num_class, features, depth + 1, criterion)
            return root
        else:
            return root


def prediction(data, root):

    if root.left is None and root.right is None:

        # res= [i for i, element in enumerate(root.bucket) if element!=0]
        # res=res[0] without pruning
        return root.labels
    else:
        if data[root.feature] < root.split_point:
            return prediction(data, root.left)
        else:
            return prediction(data, root.right)


def accuracy_calculator(test_data, test_labels, root):
    i = 0
    accuracy = 0
    pred_list = []
    for test in test_data:
        pred = prediction(test, root)
        pred = np.argmax(pred) + 1
        pred_list.append(pred)
        if pred == test_labels[i]:
            accuracy = accuracy + 1
        i = i + 1
    accuracy = accuracy / len(test_labels)
    return accuracy, np.array(pred_list)


def run_decision_tree(X_train_np, y_train_np, X_test_np, y_test_np, criterion):
    features = list(range(X_train_np.shape[1]))
    root = make_tree(X_train_np, y_train_np, NUM_CLASS, features, 0, criterion)
    accuracy, pred_list = accuracy_calculator(X_test_np, y_test_np, root)
    return accuracy, pred_list, root
