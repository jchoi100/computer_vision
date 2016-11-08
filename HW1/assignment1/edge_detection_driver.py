import os
import sys
from math import sqrt, radians, degrees, sin, cos, tan
import numpy as np
import cv2

def p5(image_in): # return edge_image_out
    """
    Finds the locations of edge points in the image.
    Uses Sobel 3 * 3 mask.
    
    Generates an edge image where the intensity at each point is
    proportional to the edge magnitude.
    """

    sobel_x = [[-1, 0, 1],\
               [-2, 0, 2],\
               [-1, 0, 1]]
    sobel_y = [[ 1, 2, 1],\
               [ 0, 0, 0],\
               [-1,-2,-1]]

    edge_image_out = image_in.copy()
    for x in range(len(image_in) - 2):
        for y in range(len(image_in[0]) - 2):
            pixel_x = (sobel_x[0][0] * image_in[x-1][y-1]) + (sobel_x[0][1] * image_in[x][y-1]) + (sobel_x[0][2] * image_in[x+1][y-1]) +\
                      (sobel_x[1][0] * image_in[x-1][y])   + (sobel_x[1][1] * image_in[x][y])   + (sobel_x[1][2] * image_in[x+1][y]) +\
                      (sobel_x[2][0] * image_in[x-1][y+1]) + (sobel_x[2][1] * image_in[x][y+1]) + (sobel_x[2][2] * image_in[x+1][y+1])
            pixel_y = (sobel_y[0][0] * image_in[x-1][y-1]) + (sobel_y[0][1] * image_in[x][y-1]) + (sobel_y[0][2] * image_in[x+1][y-1]) +\
                      (sobel_y[1][0] * image_in[x-1][y])   + (sobel_y[1][1] * image_in[x][y])   + (sobel_y[1][2] * image_in[x+1][y]) +\
                      (sobel_y[2][0] * image_in[x-1][y+1]) + (sobel_y[2][1] * image_in[x][y+1]) + (sobel_y[2][2] * image_in[x+1][y+1])

            magnitude = sqrt(pixel_x**2 + pixel_y**2)
            edge_image_out[x][y] = magnitude
    return edge_image_out

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


def p7(image_in, hough_image_in, hough_thresh): #return line_image_out
    num_rows_h = len(hough_image_in)
    num_cols_h = len(hough_image_in[0])

    lines = []
    for rho in range(num_rows_h):
        for theta in range(num_cols_h):
            if hough_image_in[rho][theta] > hough_thresh:
                lines.append((rho,theta,hough_image_in[rho][theta]))

    line_image_out = image_in.copy()
    rho_max = (num_rows_h - 1) / 2
    for (rho,theta,val) in lines:
        rho -= rho_max
        theta = radians(-theta)
        pt1 = (2**32, 2**32)
        pt2 = (2**32, 2**32)
        pt3 = (2**32, 2**32)
        pt4 = (2**32, 2**32)
        if (tan(theta) * cos(theta)) != 0:
            pt1 = (int(-rho/(tan(theta)*cos(theta))), 0)
        if cos(theta) != 0:
            pt2 = (0, int(rho/cos(theta)))
        if cos(theta) != 0 and tan(theta) != 0:
            pt3 = (int((num_rows_h - rho/cos(theta))/(tan(theta))), num_rows_h)
        if cos(theta) != 0:
            pt4 = (num_cols_h, int(tan(theta)*num_cols_h + rho/cos(theta)))
        try:
            cv2.line(line_image_out, pt1, pt2, (255,255,255), 2)
        except:
            try:
                cv2.line(line_image_out, pt1, pt3, (255,255,255), 2)
            except:
                try:
                    cv2.line(line_image_out, pt1, pt4, (255,255,255), 2)
                except:
                    try:
                        cv2.line(line_image_out, pt2, pt3, (255,255,255), 2)
                    except:
                        try:
                            cv2.line(line_image_out, pt2, pt4, (255,255,255), 2)
                        except:
                            try:
                                cv2.line(line_image_out, pt3, pt4, (255,255,255), 2)
                            except:
                                pass
    return line_image_out

def main():
    """
    Driver that:
    1. loads the images that are needed
    2. calls each of the functions
    3. reports/displays/writes the results
    """

    img = cv2.imread('hough_simple_1.pgm', 0)
    edge_image_out = p5(img)
    cv2.imwrite('hough_simple_1_p5.jpg', edge_image_out)
    [edge_image_thresh_out, hough_image_out] = p6(edge_image_out, 100)
    cv2.imwrite('hough_simple_1_thresholded.jpg', edge_image_thresh_out)
    cv2.imwrite('hough_simple_1_parameter_space.jpg', hough_image_out)
    line_image_out = p7(img, hough_image_out, 230)
    cv2.imwrite('hough_simple_1_final.jpg', line_image_out)

    img = cv2.imread('hough_simple_2.pgm', 0)
    edge_image_out = p5(img)
    [edge_image_thresh_out, hough_image_out] = p6(edge_image_out, 100)
    line_image_out = p7(img, hough_image_out, 230)
    cv2.imwrite('hough_simple_2_p5.jpg', edge_image_out)
    cv2.imwrite('hough_simple_2_thresholded.jpg', edge_image_thresh_out)
    cv2.imwrite('hough_simple_2_parameter_space.jpg', hough_image_out)
    cv2.imwrite('hough_simple_2_final.jpg', line_image_out)

    img = cv2.imread('hough_complex_1.pgm', 0)
    edge_image_out = p5(img)
    [edge_image_thresh_out, hough_image_out] = p6(edge_image_out, 100)
    line_image_out = p7(img, hough_image_out, 85)
    cv2.imwrite('hough_complex_1_p5.jpg', edge_image_out)
    cv2.imwrite('hough_complex_1_thresholded.jpg', edge_image_thresh_out)
    cv2.imwrite('hough_complex_1_parameter_space.jpg', hough_image_out)
    cv2.imwrite('hough_complex_1_final.jpg', line_image_out)

if __name__ == "__main__":
    main()
