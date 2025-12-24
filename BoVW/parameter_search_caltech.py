# open cv for SIFT
# sklearn for kmeans
# imlement BoF and knearest neighbor
import os
import cv2
import numpy as np
from sklearn import cluster
from sklearn.cluster import *
import pickle
from kNN_model import *

# from cyvlfeat.sift import *


names = os.listdir("./caltech-101/")


def Dataloader(folder):

    images = []
    gray_images = []
    dataset = []
    gray_dataset = []
    for i, dir in enumerate(names):
        images = []
        gray_images = []
        imagenames = os.listdir("./caltech-101/" + dir)
        if folder == "train":
            start_index = 0
            end_index = int(len(imagenames) * 0.8)
        elif folder == "validation":
            start_index = int(len(imagenames) * 0.8)
            end_index = len(imagenames)
        for j in range(len(imagenames[start_index:end_index])):
            path = "./caltech-101/" + dir + "/" + imagenames[j]
            im = cv2.imread(path)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            # im.show()
            images.append(im)
            gray_images.append(gray)
        dataset.append(images)
        gray_dataset.append(gray_images)

    return dataset, gray_dataset


def detect(img, first, scale, step):

    keypoints = []
    rows, cols = img.shape[:2]
    for x in range(first, rows, scale):
        for y in range(first, cols, scale):
            keypoints.append(cv2.KeyPoint(float(x), float(y), step))
    return keypoints


def DSIFT(img, first, scale, step):
    kp = detect(img, first, scale, step)
    deep_sift = cv2.SIFT_create(nfeatures=25)
    return deep_sift.compute(img, kp)


class vision_dictionary:
    def __init__(self, category, desc, hist):
        self.category = category
        self.desc = desc
        self.hist = hist

    def feature_extraction(im, gray, SIFT_type, first, scale, step):

        # nfeatures = [5,15,25,100]
        # nOctaveLayers =[3,6,9]
        # contrastThreshold = [0.4,0.7,0.9]
        # edgeThreshold = [10,12,14]
        # sigma = [1.6,2.0,2.4]

        if SIFT_type == "SIFT":
            sift = cv2.SIFT_create()
            kp = sift.detect(gray, None)
            cv2.drawKeypoints(gray, kp, im)
            kp, descrip = sift.detectAndCompute(gray, None)
        else:
            kp, descrip = DSIFT(gray, first, scale, step)
        if len(kp) == 0:
            descrip = np.reshape(np.zeros(128, "float32"), (1, 128))

        return descrip

    def all_in_one_train(kmeans_k, first, scale, step):
        from_scratch = True
        train_data, train_gray_data = Dataloader("train")
        # val_data,val_gray_data = Dataloader("validation")
        SIFT_type = ["SIFT", "D_SFIT"]
        k = [32, 64, 128, 256]
        # descriptor_all = np.ndarray((0,128),dtype="double")
        descriptor = np.ndarray((0, 128), dtype="float32")
        catalog = []
        if from_scratch:
            for j in range(len(names)):
                # descriptor = np.ndarray((0,128),dtype="double")
                for i, image in enumerate(train_data[j]):
                    new_descriptor = vision_dictionary.feature_extraction(
                        image, train_gray_data[j][i], SIFT_type[1], first, scale, step
                    )
                    new_word = vision_dictionary(names[j], new_descriptor, [])
                    catalog.append(new_word)
                    descriptor = np.concatenate((descriptor, new_descriptor))

                # descriptor_all = np.concatenate((descriptor,new_descriptor))
            # After obtain local features in descriptor call Kmeans
            clusters = MiniBatchKMeans(n_clusters=kmeans_k, random_state=0).fit(descriptor)
            clusters_center = clusters.cluster_centers_  # take cluster center -> dictionary
            pickle.dump(
                clusters, open("clusters_k_{}_probablydeep_c101.pkl".format(kmeans_k), "wb")
            )  # save clusters w:writing b:binary mode

            for j in range(len(catalog)):
                f = np.zeros(kmeans_k, dtype="float32")
                old_word = clusters.predict(catalog[j].desc)

                for w in old_word:
                    f[w] += 1
                o = np.sum(f)
                f = f / o
                catalog[j].hist = f
                # catalog[j].hist = np.histogram(old_word,bins=32,normed=True)[0]
            pickle.dump(catalog, open("catalog_k_{}_probablydeep_c101.pkl".format(kmeans_k), "wb"))
            # return catalog

        else:
            clusters = pickle.load(
                open("clusters_k_{}deep_c101.pkl".format(kmeans_k), "rb")
            )  # load clusters r:rading b:binary mode
            catalog = pickle.load(open("catalog_k_{}deep_c101.pkl".format(kmeans_k), "rb"))

    def all_in_one_test(kmeans_k, first, scale, step):
        from_scratch = True
        val_data, val_gray_data = Dataloader("validation")
        SIFT_type = ["SIFT", "D_SFIT"]
        k = [32, 64, 128, 256]
        # descriptor_all = np.ndarray((0,128),dtype="float32")
        descriptor = np.ndarray((0, 128), dtype="float32")
        catalog = []
        if from_scratch:
            for j in range(len(names)):
                # descriptor = np.ndarray((0,128),dtype="float32")
                for i, image in enumerate(val_data[j]):
                    new_descriptor = vision_dictionary.feature_extraction(
                        image, val_gray_data[j][i], SIFT_type[1], first, scale, step
                    )
                    new_word = vision_dictionary(names[j], new_descriptor, [])
                    catalog.append(new_word)
                    descriptor = np.concatenate((descriptor, new_descriptor))

                # descriptor_all = np.concatenate((descriptor,new_descriptor))
            # After obtain local features in descriptor call Kmeans
            clusters = pickle.load(open("clusters_k_{}_probablydeep_c101.pkl".format(kmeans_k), "rb"))
            clusters_center = clusters.cluster_centers_  # take cluster center -> dictionary
            # pickle.dump(clusters, open("clusters_k_{}_test_probablydeep_c101.pkl".format(kmeans_k), "wb")) #save clusters w:writing b:binary mode

            for j in range(len(catalog)):
                f = np.zeros(kmeans_k, dtype="float32")
                old_word = clusters.predict(catalog[j].desc)

                for w in old_word:
                    f[w] += 1
                o = np.sum(f)
                f = f / o
                catalog[j].hist = f
                # catalog[j].hist = np.histogram(old_word,bins=32,normed=True)[0]
            pickle.dump(catalog, open("catalog_k_{}_testdeep_c101.pkl".format(kmeans_k), "wb"))
            return catalog

        else:
            clusters = pickle.load(
                open("clusters_k_{}_testdeep_c101.pkl".format(kmeans_k), "rb")
            )  # load clusters r:rading b:binary mode
            catalog = pickle.load(open("catalog_k_{}_testdeep_c101.pkl".format(kmeans_k), "rb"))

    def GO_SVM(catalog_test):
        from sklearn import svm

        acc = 0
        catalog_train = pickle.load(open("catalog_k_{}_probably.pkl".format(k_for_kmeans), "rb"))
        X_train = []
        y_train = []
        for i in range(len(catalog_train)):
            X_train.append(catalog_train[i].hist)
            y_train.append(catalog_train[i].category)

        clf = svm.SVC(gamma="scale")
        clf.fit(X_train, y_train)

        X_test = []
        y_test = []
        for i in range(len(catalog_test)):
            X_test.append(catalog_test[i].hist)
            y_test.append(catalog_test[i].category)

        y_pred = clf.predict(X_test)

        for i in range(len(y_test)):
            if y_pred[i] == y_test[i]:
                acc += 1
        accuracy = acc / len(catalog_test)
        return accuracy

    def GO_DECISION_TREE(catalog_test, k_for_kmeans):
        from sklearn import tree

        acc = 0
        catalog_train = pickle.load(open("catalog_k_{}_probably.pkl".format(k_for_kmeans), "rb"))
        X_train = []
        y_train = []
        for i in range(len(catalog_train)):
            X_train.append(catalog_train[i].hist)
            y_train.append(catalog_train[i].category)

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)

        X_test = []
        y_test = []
        for i in range(len(catalog_test)):
            X_test.append(catalog_test[i].hist)
            y_test.append(catalog_test[i].category)

        y_pred = clf.predict(X_test)

        for i in range(len(y_test)):
            if y_pred[i] == y_test[i]:
                acc += 1
        accuracy = acc / len(catalog_test)
        return accuracy
    
    def GO_KNN(catalog_test, K_for_KNN, k_for_kmeans):

        acc = 0
        catalog_train = pickle.load(open("catalog_k_{}_probablydeep_c101.pkl".format(k_for_kmeans), "rb"))
        for i in range(len(catalog_test)):
            distance_table = calculate_distances(catalog_train, catalog_test[i])
            majority_class = majority_voting(distance_table, catalog_train, K_for_KNN)
            if majority_class == catalog_test[i].category:
                acc += 1
        accuracy = acc / len(catalog_test)
        return accuracy


if __name__ == "__main__":
    kmeans_k = 128
    knn_k = 8

    first_arr = [2, 4, 8]
    scale_arr = [2, 4, 8]
    step_arr = [4]

    # sigma=[1.6]
    for step in step_arr:  # CHANGE !!!!!
        vision_dictionary.all_in_one_train(kmeans_k, first_arr[1], scale_arr[1], step)  # CHANGE !!!!!
        catalog_test = vision_dictionary.all_in_one_test(kmeans_k, first_arr[1], scale_arr[1], step)  # CHANGE !!!!!
        accuracy = vision_dictionary.GO_KNN(catalog_test, knn_k, kmeans_k)

        with open("resultsdeep_c101.txt", "a") as f:

            f.write("scale: {}\n".format(scale_arr[1]))
            f.write("step: {}\n".format(step))
            f.write("offset: {}\n".format(first_arr[1]))
            f.write("k for KNN: {}\n".format(knn_k))
            f.write("k for kmeans: {}\n".format(kmeans_k))
            f.write("accuracy: {}\n\n".format(accuracy))

        print("scale: {}".format(scale_arr[1]))
        print("step: {}".format(step))
        print("offset: {}".format(first_arr[1]))
        print("k for KNN: {}".format(knn_k))
        print("k for kmeans: {}".format(kmeans_k))
        print("accuracy: {}".format(accuracy))
