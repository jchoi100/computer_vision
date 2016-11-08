import cv2
import matplotlib.pyplot as plt
import numpy as np

orb_results = []
orb_features = []
ret_list = []
label_list = []
center_list = []

img_number = None
if i < 10:
    img_number = '00' + str(i)
elif 10 <= i and i < 100:
    img_number = '0' + str(i)
else:
    img_number = str(i)

img = cv2.imread('./train/buildings/f000051.jpg')
orb = cv2.ORB()
keypoints = orb.detect(img, None)
keypoints, descriptors = orb.compute(img, keypoints)
orb_results.append((keypoints, descriptors))
print(descriptors.shape)
# 'descriptors is an n x 32 descriptors for n feature points in 'img'.
# whyNoSIFTinOpenCV3 = cv2.drawKeypoints(img, keypoints, 0, color=(0,255,0))
# plt.imshow(cv2.cvtColor(whyNoSIFTinOpenCV3, cv2.COLOR_BGR2RGB))
# plt.show()
for (keypoints, features) in orb_results:
    features = np.float32(features)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
    try:
        ret, label, center = cv2.kmeans(data=features, K=500, criteria=criteria, attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)
        print(ret)
        print(type(ret))
        print(label.shape)
        # print(type(label))
        print(center.shape)
        # print(type(center))
    except:
        print("=================Error occurred!(=================")
    # ret, label, center = cv2.kmeans(features, 800, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    ret_list.append(ret)
    label_list.append(label)
    center_list.append(center)








# part iii

# scene = 'buildings'
# for i in range(51, 52):
#     img_number = None
#     if i < 10:
#         img_number = '00' + str(i)
#     elif 10 <= i and i < 100:
#         img_number = '0' + str(i)
#     else:
#         img_number = str(i)
#     img = cv2.imread(PATH_TO_FILE + DATASET_TYPE[0] + '/' + scene + '/' + FILE_NAME + img_number + FILE_FORMAT)
#     orb = cv2.ORB()
#     keypoints = orb.detect(img, None)
#     keypoints, descriptors = orb.compute(img, keypoints)
#     bow_vector = []
    
#     for descriptor in descriptors:
#         min_cost = float('inf')
#         min_word = None
#         for word in bag_of_words:
#             # compute cost
#             curr_cost = euclidean(word, descriptor)
#             # curr_cost = 0.0
#             # for j in range(32):
#             #     curr_cost += (word[j] - descriptor[j])**2
#             # curr_cost = sqrt(curr_cost)
#             if curr_cost < min_cost:
#                 min_cost = curr_cost
#                 min_word = word
#         bow_vector.append(min_word)
#     bow_vectors.append(bow_vector)
#     matching_scenes.append(scene)
#     print("Finished computing encoding vector for " + PATH_TO_FILE + DATASET_TYPE[0] + '/' + scene + '/' + FILE_NAME + img_number + FILE_FORMAT + ".")
