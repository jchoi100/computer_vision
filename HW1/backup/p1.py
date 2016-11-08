import numpy as np
import cv2

def p1(grey_in, thresval): # return binary_out
    """
    Converts a grey-scale image to a binary one.
    """
    s = (len(grey_in), len(grey_in[0]))
    binary_out = np.zeros(s)
    for i in range(len(grey_in)):
        for j in range(len(grey_in[0])):
            if grey_in[i][j] > thresval:
                binary_out[i][j] = 255
            else:
                binary_out[i][j] = 0
    return binary_out
