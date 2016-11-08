import cv2
import numpy as np
from math import sqrt

El = cv2.imread('scene_l.bmp', 0)
Er = cv2.imread('scene_r.bmp', 0)
output_file = open('jchoi100_hw3_3d_point_cloud.txt', 'w')

def compute_ncc(u, v, u2):
    denom1, denom2, numer = 0.0, 0.0, 0.0
    for i in range(15):
        for j in range(15):
            denom1 += El[v + j][u + i]**2
            denom2 += Er[v + j][u2 + i]**2
            numer += (El[v + j][u + i] - Er[v + j][u2 + i])**2
    return numer / sqrt(denom1 * denom2)

b = 100.0
f = 400.0

num_rows = len(El)
num_cols = len(El[0])

three_D_points = []

for v in range(num_rows - 15):
    for u in range(num_cols - 15):
        # v is fixed even in Er
        max_ncc, max_u2 = -float('inf'), 0
        for u2 in range(num_cols - 15):
            ncc = compute_ncc(u, v, u2)
            if ncc > max_ncc:
                max_ncc, max_u2 = ncc, u2
        z = b * f / (u - max_u2)
        x = z * u / f
        y = z * v / f
        output_file.write('(' + str(x) + ', ' + str(y) + ', ' + str(z) + ')')
        three_D_points.append((x, y, z))

for (x, y, z) in three_D_points:
    print(x, y, z)


