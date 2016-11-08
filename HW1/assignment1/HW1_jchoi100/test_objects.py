import os
import sys
from p1 import p1
from p2 import p2
from p3 import p3
from p4 import p4
import cv2

def main():
    """
    Driver that:
    1. loads the images that are needed
    2. calls each of the functions
    3. reports/displays/writes the results
    """    

    #################################################################
    # Running p1, p2, p3 on all 3 sample images.
    #################################################################
    img = cv2.imread('two_objects.pgm', 0)
    binary_out = p1(img, 110)
    cv2.imwrite('two_objects_binary.jpg', binary_out)
    labels_out = p2(binary_out)
    cv2.imwrite('two_objects_labeled.jpg', labels_out)
    database_out, overlays_out = p3(labels_out)
    cv2.imwrite('two_objects_overlaid.jpg', overlays_out)

    img = cv2.imread('many_objects_1.pgm', 0)
    binary_out = p1(img, 110)
    cv2.imwrite('many_objects_1_binary.jpg', binary_out)
    labels_out = p2(binary_out)
    cv2.imwrite('many_objects_1_labeled.jpg', labels_out)
    database_out, overlays_out = p3(labels_out)
    cv2.imwrite('many_objects_1_overlaid.jpg', overlays_out)
            
    img = cv2.imread('many_objects_2.pgm', 0)
    binary_out = p1(img, 110)
    cv2.imwrite('many_objects_2_binary.jpg', binary_out)
    labels_out = p2(binary_out)
    cv2.imwrite('many_objects_2_labeled.jpg', labels_out)
    database_out, overlays_out = p3(labels_out)
    cv2.imwrite('many_objects_2_overlaid.jpg', overlays_out)

    #################################################################
    # Running p4 using two_images.pgm as model and many_objects_1,2
    # as sample test images.
    #################################################################
    two_objects_img = cv2.imread('two_objects.pgm', 0)
    many_objects_1_img = cv2.imread('many_objects_1.pgm', 0)
    many_objects_2_img = cv2.imread('many_objects_2.pgm', 0)

    labels = p2(p1(cv2.imread('two_objects.pgm', 0), 110))
    database_out, _ = p3(labels)
    labels_test = p2(p1(cv2.imread('many_objects_1.pgm', 0), 110))
    overlays_out = p4(labels_test, database_out)
    cv2.imwrite('many_objects_1_detection.jpg', overlays_out)

    labels = p2(p1(cv2.imread('two_objects.pgm', 0), 110))
    database_out, _ = p3(labels)
    labels_test = p2(p1(cv2.imread('many_objects_2.pgm', 0), 110))
    overlays_out = p4(labels_test, database_out)
    cv2.imwrite('many_objects_2_detection.jpg', overlays_out)

if __name__ == "__main__":
    main()
