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
