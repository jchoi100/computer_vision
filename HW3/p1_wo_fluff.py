import cv2, sys, pickle
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from scipy.spatial.distance import euclidean
from scipy.cluster.vq import whiten
from scipy.spatial import cKDTree

SCENE_TYPE = ['buildings', 'cars', 'food', 'people', 'trees']
orb = cv2.ORB()

# i, ii) Collect orb features for all training images into one large vector and run kmeans
orb_features = []
for scene in SCENE_TYPE:
    try:
        scene_descriptors = pickle.load(open(scene + '_descriptors.p', 'rb'))
    except:
        scene_descriptors = []
        for i in range(51, 201):
            if i < 10:
                img_number = '00' + str(i)
            elif 10 <= i and i < 100:
                img_number = '0' + str(i)
            else:
                img_number = str(i)
            img = cv2.imread('./train/' + scene + '/f000' + img_number + '.jpg')
            keypoints = orb.detect(img, None)
            keypoints, descriptors = orb.compute(img, keypoints)
            try:
                scene_descriptors.extend(descriptors)
            except:
                pass
        pickle.dump(scene_descriptors, open(scene + '_descriptors.p', 'wb'))
    orb_features.extend(scene_descriptors)
# print("Finished computing orb features for all images!")

features = whiten(np.float32(orb_features))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K_list = [800]
query_list = [9, 11, 13, 15]

for k in K_list:
    for q in query_list:
        # try:
        #     bag_of_words = pickle.load(open('bag_of_words.p', 'rb'))
        # except:
        try:
            ret, label, bag_of_words = cv2.kmeans(data=features, K=k, criteria=criteria,\
                                                  attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
            # pickle.dump(bag_of_words, open('bag_of_words.p', 'wb'))  
        except:
            sys.exit(1)
        bag_of_words = cKDTree(bag_of_words)
        # print("Finished computing kmeans!")

        # iii) Create BoW encoding vector for each training image using BoW
        bow_vectors, matching_scenes = [], []
        for scene in SCENE_TYPE:
            for i in range(51, 201):
                if i < 10:
                    img_number = '00' + str(i)
                elif 10 <= i and i < 100:
                    img_number = '0' + str(i)
                else:
                    img_number = str(i)
                img = cv2.imread('./train/' + scene + '/f000' + img_number + '.jpg')
                keypoints = orb.detect(img, None)
                keypoints, descriptors = orb.compute(img, keypoints)
                bow_vector = np.array([0.] * k)
                try:
                    for descriptor in descriptors:
                        dist, match = bag_of_words.query(descriptor, q)
                        for j in range(len(match)):
                            bow_vector[match[j]] = 1.0 / (dist[j]**2 + 1)
                except:
                    pass
                    # print("----> Error in computing encoding vector for ./train/" + scene + "/f000" + img_number + ".jpg")
                bow_vectors.append((bow_vector - np.mean(bow_vector)) / np.std(bow_vector))
                matching_scenes.append(scene)
        # print("Finished computing encoding vector for all training images.")

        # iv) Match testing image with label and check accuracy
        bow_vectors = cKDTree(bow_vectors)
        num_correct, num_images = 0.0, 0.0
        for scene in SCENE_TYPE:
            for i in range(51):
                if i < 10:
                    img_number = '00' + str(i)
                elif 10 <= i and i < 100:
                    img_number = '0' + str(i)
                else:
                    img_number = str(i)
                img = cv2.imread('./test/' + scene + '/f000' + img_number + '.jpg')
                num_images += 1.0
                keypoints = orb.detect(img, None)
                keypoints, descriptors = orb.compute(img, keypoints)
                bow_vector = np.array([0.] * k)
                try:
                    for descriptor in descriptors:
                        dist, match = bag_of_words.query(descriptor, q)
                        for j in range(len(match)):
                            bow_vector[match[j]] = 1.0 / (dist[j]**2 + 1)
                    bow_vector = (bow_vector - np.mean(bow_vector)) / np.std(bow_vector) # Normalize
                except:
                    pass
                    # print("----> Error in computing encoding vector for ./test/" + scene + "/f000" + img_number + ".jpg")
                dist, match = bow_vectors.query(bow_vector) # 1 Nearest Neighbor search to find best match
                min_scene = matching_scenes[match]
                if min_scene == scene:
                    num_correct += 1.0
        # print("Finished computing encoding vector for all test images.")

        print("============ Accuracy ============")
        print("k = " + str(k) + ", q = " + str(q) + " : " + str(num_correct / num_images) + " (" + str(num_correct) + " / " + str(num_images) + ")")
        print("==================================")