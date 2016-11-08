import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from scipy.spatial.distance import euclidean

PATH_TO_FILE = './'
DATASET_TYPE = ['train', 'test']
SCENE_TYPE = ['buildings', 'cars', 'faces', 'food', 'people', 'trees']
FILE_NAME = 'f000'
FILE_FORMAT = '.jpg'

orb_features = []

print("============================= Part i, ii =============================")
# i, ii) Collect orb features for all training images into one large vector and run kmeans
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
            orb_features.extend(descriptors)
        except:
            pass
        # print("Finished computing orb for " + PATH_TO_FILE + DATASET_TYPE[0] + '/' + \
        #         scene + '/' + FILE_NAME + img_number + FILE_FORMAT + ".")
    print("----------------------------------------------------------------------------")
    print("Finished computing orb for all images in training class " + scene + ".")
    print("----------------------------------------------------------------------------")

features = np.float32(orb_features)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

try:
    print("Computing kmeans. This can take a while...")
    ret, label, bag_of_words = cv2.kmeans(data=features, K=800, criteria=criteria, \
                                            attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
    print("Successful!")
except:
    print("Failed!")
    sys.exit(1)

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
        img = cv2.imread(PATH_TO_FILE + DATASET_TYPE[0] + '/' + scene + \
                '/' + FILE_NAME + img_number + FILE_FORMAT)
        orb = cv2.ORB()
        keypoints = orb.detect(img, None)
        keypoints, descriptors = orb.compute(img, keypoints)
        bow_vector = {}
        for j in range(len(bag_of_words)):
            bow_vector[j] = 0.0
        sum_weights = 0.0
        try:
            for descriptor in descriptors:
                min_cost = float('inf')
                min_word = None
                min_index = -1
                for j in range(len(bag_of_words)):
                    curr_cost = euclidean(bag_of_words[j], descriptor)
                    if curr_cost < min_cost:
                        min_cost = curr_cost
                        min_word = bag_of_words[j]
                        min_index = j
                bow_vector[min_index] += 1 / (min_cost**2 + 1)
                sum_weights += 1 / (min_cost**2 + 1)
            # Normalize
            for key, value in bow_vector.items():
                bow_vector[key] = value / float(sum_weights)
        except:
            print("             Error in computing encoding vector for " + \
                    PATH_TO_FILE + DATASET_TYPE[0] + '/' + scene + '/' + \
                    FILE_NAME + img_number + FILE_FORMAT + "!")

        bow_vectors.append(bow_vector)
        matching_scenes.append(scene)
        # print("Finished computing encoding vector for " + PATH_TO_FILE + DATASET_TYPE[0] + \
        #         '/' + scene + '/' + FILE_NAME + img_number + FILE_FORMAT + ".")
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
        img = cv2.imread(PATH_TO_FILE + DATASET_TYPE[1] + '/' + scene + \
                            '/' + FILE_NAME + img_number + FILE_FORMAT)
        num_images += 1
        orb = cv2.ORB()
        keypoints = orb.detect(img, None)
        keypoints, descriptors = orb.compute(img, keypoints)
        bow_vector = {}
        for j in range(len(bag_of_words)):
            bow_vector[j] = 0.0
        sum_weights = 0.0
        try:
            for descriptor in descriptors:
                min_cost = float('inf')
                min_word = None
                min_index = -1
                for j in range(len(bag_of_words)):
                    curr_cost = euclidean(bag_of_words[j], descriptor)
                    if curr_cost < min_cost:
                        min_cost = curr_cost
                        min_word = bag_of_words[j]
                        min_index = j
                bow_vector[min_index] += 1 / (min_cost**2 + 1)
                sum_weights += 1 / (min_cost**2 + 1)
            # Normalize
            for key, value in bow_vector.items():
                bow_vector[key] = bow_vector[key] / sum_weights
        except:
            print("             Error in computing encoding vector for " + \
                    PATH_TO_FILE + DATASET_TYPE[1] + '/' + scene + '/' + \
                    FILE_NAME + img_number + FILE_FORMAT + "!")
        
        # print("Finished computing encoding vector for " + PATH_TO_FILE + DATASET_TYPE[1] + \
        #         '/' + scene + '/' + FILE_NAME + img_number + FILE_FORMAT + ".")

        min_cost = float('inf')
        min_scene = None
        # Use 1-NN search to the training features to get best label
        for j in range(len(bow_vectors)):
            curr_cost = euclidean(bow_vectors[j].values(), bow_vector.values())
            if curr_cost < min_cost:
                min_cost = curr_cost
                min_scene = matching_scenes[j]
        if min_scene == scene:
            num_correct += 1
    print("----------------------------------------------------------------------------")
    print("Finished computing encoding vector for all test images in " + scene + ".")
    print("----------------------------------------------------------------------------")

print("============ Accuracy ============")
print(str(num_correct / num_images) + " (" + str(num_correct) + " / " + str(num_images) + ")")
print("==================================")
