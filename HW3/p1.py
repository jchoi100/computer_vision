import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from scipy.spatial.distance import euclidean
from scipy.cluster.vq import whiten
from scipy.spatial import cKDTree
import pdb
import pickle

PATH_TO_FILE = './'
DATASET_TYPE = ['train', 'test']
SCENE_TYPE = ['buildings', 'cars', 'faces', 'food', 'people', 'trees']
FILE_NAME = 'f000'
FILE_FORMAT = '.jpg'

print("============================= Part i, ii =============================")
# i, ii) Collect orb features for all training images into one large vector and run kmeans
orb_features = []

for scene in SCENE_TYPE:
    for i in range(51, 200):
        img_number = None
        if i < 10:
            img_number = '00' + str(i)
        elif 10 <= i and i < 100:
            img_number = '0' + str(i)
        else:
            img_number = str(i)
        img = cv2.imread(PATH_TO_FILE + DATASET_TYPE[0] + '/' + scene + '/' + \
                            FILE_NAME + img_number + FILE_FORMAT)
        orb = cv2.ORB()
        keypoints = orb.detect(img, None)
        keypoints, descriptors = orb.compute(img, keypoints)
        try:
            orb_features.extend(descriptors) #############################################################This might be append, not extend###################
        except:
            pass
    print("----------------------------------------------------------------------------")
    print("Finished computing orb for all images in training class " + scene + ".")
    print("----------------------------------------------------------------------------")

features = np.float32(orb_features)
features = whiten(features)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

try:
    bag_of_words = pickle.load(open('bag_of_words.p', 'rb'))
except:
    try:
        print("Computing kmeans. This can take a while...")
        ret, label, bag_of_words = cv2.kmeans(data=features, K=800, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
        pickle.dump(bag_of_words, open('bag_of_words.p', 'wb'))  
        print("Successful!")
    except:
        print("Failed!")
        sys.exit(1)  

bag_of_words = cKDTree(bag_of_words)
bow_vectors = []
matching_scenes = []

print("============================= Part iii =============================")
# iii) Create BoW encoding vector for each training image using BoW
for scene in SCENE_TYPE:
    for i in range(51, 200):
        img_number = None
        if i < 10:
            img_number = '00' + str(i)
        elif 10 <= i and i < 100:
            img_number = '0' + str(i)
        else:
            img_number = str(i)
        img = cv2.imread(PATH_TO_FILE + DATASET_TYPE[0] + '/' + scene + '/' + FILE_NAME + img_number + FILE_FORMAT)
        orb = cv2.ORB()
        keypoints = orb.detect(img, None)
        keypoints, descriptors = orb.compute(img, keypoints)
        bow_vector = np.array([0.]*800)
        try:
            for descriptor in descriptors:
                dist, match = bag_of_words.query(descriptor, 5)
                for i in range(len(match)):
                    bow_vector[match[i]] = 1 / dist[i]
            # Normalize
            std = np.std(bow_vector) if np.std(bow_vector) != 0 else 0.001
            bow_vectors.append((bow_vector - np.mean(bow_vector)) / std)
            matching_scenes.append(scene)
        except:
            print("      Error in computing encoding vector for " + \
                    PATH_TO_FILE + DATASET_TYPE[0] + '/' + scene + '/' + \
                    FILE_NAME + img_number + FILE_FORMAT + "!")

    print("----------------------------------------------------------------------------")
    print("Finished computing encoding vector for all training images in " + scene + ".")
    print("----------------------------------------------------------------------------")

print("============================= Part iv =============================")
# iv) Match testing image with label and check accuracy
num_correct = 0.0
num_images = 0.0
for scene in SCENE_TYPE:
    for i in range(51):
        img_number = None
        if i < 10:
            img_number = '00' + str(i)
        elif 10 <= i and i < 100:
            img_number = '0' + str(i)
        else:
            img_number = str(i)
        img = cv2.imread(PATH_TO_FILE + DATASET_TYPE[1] + '/' + scene + '/' + FILE_NAME + img_number + FILE_FORMAT)
        num_images += 1
        orb = cv2.ORB()
        keypoints = orb.detect(img, None)
        keypoints, descriptors = orb.compute(img, keypoints)
        bow_vector = np.array([0.]*800)
        try:
            for descriptor in descriptors:
                dist, match = bag_of_words.query(descriptor, 5)
                for i in range(len(match)):
                    bow_vector[match[i]] = 1 / dist[i]
            # Normalize
            std = np.std(bow_vector) if np.std(bow_vector) != 0 else 0.001
            bow_vector = (bow_vector - np.mean(bow_vector)) / std
        except:
            print("      Error in computing encoding vector for " + \
                    PATH_TO_FILE + DATASET_TYPE[1] + '/' + scene + '/' + \
                    FILE_NAME + img_number + FILE_FORMAT + "!")

        print(bag_of_words.shape)
        print(bag_of_words[0].shape)
        print(bow_vector.shape)

        # Use 1-NN search to the training features to get best label
        dist, match = bag_of_words.query(bow_vector, 1)
        min_scene = matching_scenes[match[0]]
        if min_scene == scene:
            num_correct += 1
    print("----------------------------------------------------------------------------")
    print("Finished computing encoding vector for all test images in " + scene + ".")
    print("----------------------------------------------------------------------------")

print("============ Accuracy ============")
print(str(num_correct / num_images) + " (" + str(num_correct) + " / " + str(num_images) + ")")
print("==================================")
