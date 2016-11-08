from math import sqrt, radians, sin, cos
import numpy as np

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
