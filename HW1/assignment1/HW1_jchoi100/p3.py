import numpy as np
import cv2
from math import sin, cos, atan2, degrees, pi, tan, radians, sqrt

def p3(labels_in): # retrun [database_out, overlays_out]
    """
    Each object contains at least the following attributes.

    "object_label", the index label of the object
    "x_position" of the center 
    "y_position" of the center
    "min_moment", the minimum moment of inertia
    "orientation", the angle (in DEGREES!) between the axis
                    of minimum inertia and vertical axis
    "roundness", the roundness of the object
    """
    database_out = [] # list of dictonaries
    overlays_out = labels_in
    labels = []
    num_rows = len(labels_in)
    num_cols = len(labels_in[0])

    for i in range(num_rows):
        for j in range(num_cols):
            if labels_in[i][j] != 0 and labels_in[i][j] not in labels:
                labels.append(labels_in[i][j])
                    
    for label in labels:
        curr_object = {}
        area = 0.0
        # Compute area
        for i in range(num_rows):
            for j in range(num_cols):
                if labels_in[i][j] == label:
                    area += 1

        x_position = 0 # center point
        y_position = 0 # center point
        a_prime = 0 # second moment
        b_prime = 0 # second moment
        c_prime = 0 # second moment

        for i in range(num_rows):
            for j in range(num_cols):
                b_ij = 0
                if labels_in[i][j] == label:
                    b_ij = 1
                x_position += (j * b_ij / area)
                y_position += (i * b_ij / area)
                a_prime += (i**2 * b_ij)
                b_prime += (2 * i * j * b_ij)
                c_prime += (j**2 * b_ij)
        
        a = a_prime - (y_position**2) * area
        b = b_prime - 2 * x_position * y_position * area
        c = c_prime - (x_position**2) * area

        theta1 = atan2(b, a - c) / 2
        theta2 = theta1 + pi / 2
        
        e_min = a * (sin(theta1)**2) - b * sin(theta1) * cos(theta1) + c * (cos(theta1)**2)
        e_max = a * (sin(theta2)**2) - b * sin(theta2) * cos(theta2) + c * (cos(theta2)**2)
        roundness = e_min / e_max

        curr_object['object_label'] = label
        curr_object['x_position'] = x_position
        curr_object['y_position'] = y_position
        curr_object['min_moment'] = e_min
        curr_object['orientation'] = 90 - degrees(theta1)
        curr_object['roundness'] = roundness

        database_out.append(curr_object)

        slope = tan(radians(90 - degrees(theta1)))
        y_intercept = y_position - slope * x_position

        q_a = (slope**2 + 1)
        q_b = (-2 * x_position + 2 * slope * (y_intercept - y_position))
        q_c = x_position**2 + (y_intercept - y_position)**2 - 400
        
        x_2 = (-q_b + sqrt(q_b**2 - 4 * q_a * q_c)) / (2 * q_a)
        if x_2 < 0 or x_2 > len(labels_in[0]):
            x_2 = (-q_b - sqrt(q_b**2 - 4 * q_a * q_c)) / (2 * q_a)
        y_2 = slope * x_2 + y_intercept

        cv2.circle(overlays_out, (int(x_position), int(y_position)), 5, (0,255,255), 1) # circle around the center
        cv2.line(overlays_out, (int(x_position), int(y_position)), (int(x_2), int(y_2)), (100,100,100), 2) # Draws a line connecting the center and another point on e_min
    return database_out, overlays_out
