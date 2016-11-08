from math import sqrt, radians, sin, cos
import numpy as np

def p6(edge_image_in, edge_thresh): #return [edge_image_thresh_out, hough_image_out]
    """
    Generate thresholded image and hough transformed image.
    """

    num_rows = len(edge_image_in)
    num_cols = len(edge_image_in[0])

    # Thresholded image portion.
    edge_image_thresh_out = edge_image_in.copy()
    for y in range(num_rows):
        for x in range(num_cols):
            if edge_image_in[y][x] >= edge_thresh:
                edge_image_thresh_out[y][x] = 255
            else:
                edge_image_thresh_out[y][x] = 0

    # Hough transformed image portion.
    hough_image_out = edge_image_thresh_out.copy()
    rho_max = int(sqrt(num_rows**2 + num_cols**2))
    theta_max = 90
    A = np.zeros((int(rho_max * 2 + 1), int(theta_max * 2 + 1))) # A[rho][theta]
    for y in range(num_rows):
        for x in range(num_cols):
            if hough_image_out[y][x] == 255:
                for theta in range(-theta_max, theta_max + 1):
                    rho = int(x * cos(radians(theta)) + y * sin(radians(theta)))
                    A[int((rho + rho_max))][int((theta + theta_max))] += 1
    hough_image_out = A
    return [edge_image_thresh_out, hough_image_out]

###################################
# Pseudocode for Hough Transform
# 1. Partition the rho-th plane into accumulator cells A[rho, th], where
#    rho \in [-norm(img size), norm(img size)] and th \in [-90 deg, 90 deg]
# 2. The cell (i,j) corresponds to the square associated with parameter values
#    (th_j, rho_i)
# 3. Init all cells with value 0
# 4. For each foreground point (x_k, y_k) in the thresholded edge image,
#    - Let th_j equal all the possible th values.
#         - solve for rho using rho = xcos(th_j) + ysin(th_j)
#         - round rho to the closest cell value, rho_q
#         - Imcrement A(i,q) if the th_j results in rho_q
# 5. After this procedure, A(i,j)=P means that P points in the xy-space lie
#    on the line rho_j = xcos(th_j) + ysin(th_j)
# 6. Find line candidates where A(i,j) is above a suitable threshold value.
