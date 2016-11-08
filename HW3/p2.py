import cv2
import numpy as np
from math import sqrt

WINDOW_SIZE = 15
b = 100
f = 400

img_l = cv2.imread('scene_l.bmp', 0)
img_r = cv2.imread('scene_r.bmp', 0)

num_rows = len(img_l)
num_cols = len(img_l[0])

# matches = np.zeros(img_l.shape)
# x_coords = np.zeros(img_l.shape)
# y_coords = np.zeros(img_l.shape)
z_coords = np.zeros(img_l.shape)

min_ncc = float('inf')
k, l = -1, -1


for i in range(num_rows - WINDOW_SIZE):
    for j in range(num_cols - WINDOW_SIZE):
        # We have img_l's i, j in our hands.
        max_ncc, max_j2 = -float('inf'), -1
        for j2 in range(num_cols - WINDOW_SIZE):
            numerator = 0.0
            for k1 in range(WINDOW_SIZE):
                for k2 in range(WINDOW_SIZE):
                    numerator += (img_l[i + k1][j + k2] - img_r[i + k1][j2 + k2])**2
            denom_elem1 = 0.0
            for k1 in range(WINDOW_SIZE):
                for k2 in range(WINDOW_SIZE):
                    denom_elem1 += (img_l[i + k1][j + k2])**2
            denom_elem2 = 0.0
            for k1 in range(WINDOW_SIZE):
                for k2 in range(WINDOW_SIZE):
                    denom_elem2 += (img_r[i + k1][j2 + k2])**2
            # print denom_elem1
            # print denom_elem2
            denominator = sqrt(denom_elem1 * denom_elem2)
            curr_ncc = numerator / denominator
            if curr_ncc > max_ncc:
                max_ncc, max_j2 = curr_ncc, j2
        disparity = max_ncc
        z_coords[i][j] = b * f / disparity

print(z_coords)


# for i in range(num_rows):
    # for j in range(num_cols):
        # print(z_coords[i][j])



