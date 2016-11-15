import cv2
import numpy as np
from math import sqrt

El, Er = cv2.imread('scene_l.bmp', 0), cv2.imread('scene_r.bmp', 0)
b, f = 100.0, 400.0
output_file = open('jchoi100_hw3_3d_point_cloud.txt', 'w')
LEFT_MARGIN_BUFFER = 100
WINDOW_SIZE = 15

def compute_ncc(u, v, u2):
    denom1, denom2, numer = 0.0, 0.0, 0.0
    for i in range(WINDOW_SIZE):
        for j in range(WINDOW_SIZE):
            denom1 += El[v + j][u + i]**2
            denom2 += Er[v + j][u2 + i]**2
            numer += (El[v + j][u + i] * Er[v + j][u2 + i])
    return numer / (sqrt(denom1) * sqrt(denom2))

depth_map = np.zeros(El.shape)
num_rows, num_cols = len(El), len(El[0])

print("num_rows: " + str(num_rows))
print("num_cols: " + str(num_cols))
print("------------------")

for v in range(num_rows - WINDOW_SIZE):
    for u in range(num_cols - WINDOW_SIZE):
        # v is fixed even in Er
        max_ncc, max_u2 = -float('inf'), 0.0
        for u2 in range(u):
            ncc = compute_ncc(u, v, u2)
            if ncc > max_ncc:
                max_ncc, max_u2 = ncc, u2
        try:
            z = b * f / (u - max_u2)
            x = z * u / f
            y = z * v / f
        except:
            print("   ---> There was no disparity! Writing (-1, -1, -1) to output file...")
            x, y, z = -1, -1, -1
            depth_map[v][u] = z
        output_file.write('(' + str(x) + ', ' + str(y) + ', ' + str(z) + ')\n')
    print("Finished computing row" + str(v) + ". Still " + str(num_rows - WINDOW_SIZE - v) + " rows left...")

for v in range(num_rows):
    for u in range(num_cols):
        # Handling pixels with disparity 0
        if depth_map[v][u] == -1:
            # Handling edge cases and then normal case in the end.
            if v == 0 and u == 0:
                depth_map[v][u] = (depth_map[v+1][u] + depth_map[v][u+1]) / 2
            elif v == 0 and u == num_cols - 1:
                depth_map[v][u] = (depth_map[v+1][u] + depth_map[v][u-1]) / 2
            elif v == num_rows - 1 and u == 0:
                depth_map[v][u] = (depth_map[v-1][u] + depth_map[v][u+1]) / 2
            elif v == num_rows - 1 and u == num_cols - 1:
                depth_map[v][u] = (depth_map[v-1][u] + depth_map[v][u-1]) / 2
            elif v == 0:
                depth_map[v][u] = (depth_map[v][u-1] + depth_map[v][u+1] + depth_map[v+1][u]) / 3
            elif u == 0:
                depth_map[v][u] = (depth_map[v-1][u] + depth_map[v+1][u] + depth_map[v][u+1]) / 3
            elif v == num_rows - 1:
                depth_map[v][u] = (depth_map[v][u-1] + depth_map[v][u+1] + depth_map[v-1][u]) / 3
            elif u == num_cols - 1:
                depth_map[v][u] = (depth_map[v+1][u] + depth_map[v-1][u] + depth_map[v][u-1]) / 3
            else:
                depth_map[v][u] = (depth_map[v-1][u] + depth_map[v+1][u] + depth_map[v][u-1] + depth_map[v][u+1]) / 4

# Normalize/rescale to 0~255
depth_map *= (255.0/depth_map.max())
print(depth_map)
print(depth_map.max())
print(depth_map.min())
cv2.imwrite('jchoi100_hw3_depth_map.jpg', depth_map)