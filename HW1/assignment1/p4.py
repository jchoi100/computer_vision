import numpy as np
import cv2
from math import sin, cos, atan2, degrees, pi, tan, radians, sqrt
from p3 import p3

def p4(labels_in, database_in): # return overlays_out
    """
    Compares the attributes of each object in a labeled image file
    "labels_in" with those from the object model database "database_in".

    Produces an output image that displays the positions and orientations
    (using circles and line segements as in p3.py) of only those objects
    that have been recognized.

    Using the database generated from "two_objects.pgm", test this function
    on the images "many_objects_1.pgm" and "many_objects_2.pgm".

    In the README file, state the comparison criteria and thresholds used.
    """
    """
    A dict item in a database list contains:
    "object_label", the index label of the object
    "x_position" of the center 
    "y_position" of the center
    "min_moment", the minimum moment of inertia
    "orientation", the angle (in DEGREES!) between the axis
                    of minimum inertia and vertical axis
    "roundness", the roundness of the object
    """
    overlays_out = labels_in.copy()
    database_test, overlays = p3(labels_in)

    # Iterate through each object in the newly seen image. e.g. many_objects_1.pgm
    for item in database_test:
        orientation = item['orientation']
        roundness = item['roundness']
        x_position = item['x_position']
        y_position = item['y_position']
        for model_item in database_in:
            model_roundness = model_item['roundness']
            if model_roundness * 0.9 <= roundness and roundness <= model_roundness * 1.1:
                slope = tan(radians(orientation))
                y_intercept = y_position - slope * x_position

                q_a = (slope**2 + 1)
                q_b = (-2 * x_position + 2 * slope * (y_intercept - y_position))
                q_c = x_position**2 + (y_intercept - y_position)**2 - 400
                
                x_2 = (-q_b + sqrt(q_b**2 - 4 * q_a * q_c)) / (2 * q_a)
                if x_2 < 0 or x_2 > len(overlays[0]):
                    x_2 = (-q_b - sqrt(q_b**2 - 4 * q_a * q_c)) / (2 * q_a)
                y_2 = slope * x_2 + y_intercept

                cv2.circle(overlays_out, (int(x_position), int(y_position)), 5, (0,255,255), 1) # circle around the center
                cv2.line(overlays_out, (int(x_position), int(y_position)), (int(x_2), int(y_2)), (100,100,100), 2) # Draws a line connecting the center and another point on e_min
    return overlays_out
