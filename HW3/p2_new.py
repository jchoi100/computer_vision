import cv2
import numpy as np
from math import sqrt

El, Er = cv2.imread('scene_l.bmp', 0), cv2.imread('scene_r.bmp', 0)
b, f = 100.0, 400.0
output_file = open('jchoi100_hw3_3d_point_cloud.txt', 'w')
LEFT_MARGIN_BUFFER = 100
WINDOW_SIZE = 15
depth_map = np.zeros(El.shape)
num_rows, num_cols = len(El), len(El[0])

def compute_ncc(u, v, u2):
    l_window = np.float32(El[v:v+WINDOW_SIZE, u:u+WINDOW_SIZE].flatten())
    r_window = np.float32(Er[v:v+WINDOW_SIZE, u2:u2+WINDOW_SIZE].flatten())
    l_window -= np.mean(l_window)
    r_window -= np.mean(r_window)
    l_window /= np.std(l_window)
    r_window /= np.std(r_window)
    return np.dot(l_window, r_window)

for v in range(num_rows - WINDOW_SIZE):
    for u in range(num_cols - WINDOW_SIZE):
        # v is fixed even in Er
        max_ncc, max_u2 = -float('inf'), 0.0
        if u > 14:
            for u2 in range(u - 14, u):
                ncc = compute_ncc(u, v, u2)
                if ncc > max_ncc:
                    max_ncc, max_u2 = ncc, u2
        else:
            max_u2 = -1
        try:
            z = b * f / (u - max_u2)
            x = z * u / f
            y = z * v / f
        except:
            # No disparity? (i.e. u - max_u2 == 0 case)
            x, y, z = -1, -1, -1
        depth_map[v][u] = z
        output_file.write('(' + str(x) + ', ' + str(y) + ', ' + str(z) + ')\n')
    # print("Finished computing row" + str(v) + ". Still " + str(num_rows - WINDOW_SIZE - v - 1) + " rows left...")

# Interpolate empty holes.
for v in range(num_rows):
    for u in range(num_cols):
        # Handling pixels with disparity 0
        if depth_map[v][u] == -1:
            # Handling edge cases and then normal case in the end.
            if v == 0 and u == 0:
                depth_map[v][u] = (depth_map[v+1][u] + depth_map[v][u+1]) / 2.0
            elif v == 0 and u == num_cols - 1:
                depth_map[v][u] = (depth_map[v+1][u] + depth_map[v][u-1]) / 2.0
            elif v == num_rows - 1 and u == 0:
                depth_map[v][u] = (depth_map[v-1][u] + depth_map[v][u+1]) / 2.0
            elif v == num_rows - 1 and u == num_cols - 1:
                depth_map[v][u] = (depth_map[v-1][u] + depth_map[v][u-1]) / 2.0
            elif v == 0:
                depth_map[v][u] = (depth_map[v][u-1] + depth_map[v][u+1] + depth_map[v+1][u]) / 3.0
            elif u == 0:
                depth_map[v][u] = (depth_map[v-1][u] + depth_map[v+1][u] + depth_map[v][u+1]) / 3.0
            elif v == num_rows - 1:
                depth_map[v][u] = (depth_map[v][u-1] + depth_map[v][u+1] + depth_map[v-1][u]) / 3.0
            elif u == num_cols - 1:
                depth_map[v][u] = (depth_map[v+1][u] + depth_map[v-1][u] + depth_map[v][u-1]) / 3.0
            else:
                depth_map[v][u] = (depth_map[v-1][u] + depth_map[v+1][u] + depth_map[v][u-1] + depth_map[v][u+1]) / 4.0

# Normalize/rescale to 0~255
depth_map *= (255.0/depth_map.max())
cv2.imwrite('jchoi100_hw3_depth_map.jpg', depth_map)