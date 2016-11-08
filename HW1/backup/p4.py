import numpy as np
import cv2
from math import sin, cos, atan2, degrees, pi, tan, radians, sqrt

"""UnionFind.py

Union-find data structure. Based on Josiah Carlson's code,
http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/215912
with significant additional changes by D. Eppstein.
"""

class UnionFind:
    """Union-find data structure.

    Each unionFind instance X maintains a family of disjoint sets of
    hashable objects, supporting the following two methods:

    - X[item] returns a name for the set containing the given item.
      Each set is named by an arbitrarily-chosen one of its members; as
      long as the set remains unchanged it will keep the same name. If
      the item is not yet part of a set in X, a new singleton set is
      created for it.

    - X.union(item1, item2, ...) merges the sets containing each item
      into a single larger set.  If any item is not yet part of a set
      in X, it is added to X as one of the members of the merged set.
    """

    def __init__(self):
        """Create a new empty union-find structure."""
        self.weights = {}
        self.parents = {}

    def __getitem__(self, object):
        """Find and return the name of the set containing the object."""

        # check for previously unknown object
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root
        
    def __iter__(self):
        """Iterate through all items ever found or unioned by this structure."""
        return iter(self.parents)

    def union(self, *objects):
        """Find the sets containing the objects and merge them all."""
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r],r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest


def p1(grey_in, thresval): # return binary_out
    s = (len(grey_in), len(grey_in[0]))
    binary_out = np.zeros(s)
    for i in range(len(grey_in)):
        for j in range(len(grey_in[0])):
            if grey_in[i][j] > thresval:
                binary_out[i][j] = 255
            else:
                binary_out[i][j] = 0
    return binary_out


def p2(binary_in): # return labels_out
    """
    Sequential labeling algorithm that segments
    a binary image into several connected regions.
    """
    label_count = 1
    num_rows = len(binary_in)
    num_cols = len(binary_in[0])
    s = (num_rows, num_cols)
    labels_out = np.zeros(s)
    equivalence_relationships = UnionFind()
    for i in range(num_rows):
        for j in range(num_cols):
            curr = binary_in[i][j]
            nw = binary_in[i - 1][j - 1]
            n = binary_in[i - 1][j]
            w = binary_in[i][j - 1]

            if curr == 0:
                labels_out[i][j] = 0
            elif nw == 0 and n == 0 and w == 0:
                label_count += 1
                labels_out[i][j] = label_count
                equivalence_relationships[label_count] # create new set
            elif labels_out[i - 1][j - 1] != 0:
                labels_out[i][j] = labels_out[i - 1][j - 1]
            elif nw == 0 and n == 0 and labels_out[i][j - 1] != 0:
                labels_out[i][j] = labels_out[i][j - 1]
            elif nw == 0 and w == 0 and labels_out[i - 1][j] != 0:
                labels_out[i][j] = labels_out[i - 1][j]
            elif nw == 0 and labels_out[i][j - 1] != 0 and labels_out[i - 1][j] != 0:
                labels_out[i][j] = labels_out[i - 1][j]
                set_a = equivalence_relationships[labels_out[i][j - 1]]
                set_b = equivalence_relationships[labels_out[i - 1][j]]
                if set_a != set_b:
                    equivalence_relationships.union(equivalence_relationships[labels_out[i][j - 1]],\
                                                    equivalence_relationships[labels_out[i - 1][j]])
    new_label_dict = {}
    # print(equivalence_relationships)
    start_label = 70
    count = 1
    offset = 30
    for i in range(num_rows):
        for j in range(num_cols):
            label = 0
            curr = labels_out[i][j]
            curr_set = equivalence_relationships[curr]
            if curr == 0:
                labels_out[i][j] = 0
            else:
                if not new_label_dict.has_key(curr_set):
                    count += 1
                    new_label_dict[curr_set] = start_label + count * offset
                    # new_label_dict[curr_set] = count
                labels_out[i][j] = new_label_dict[curr_set]
    return labels_out


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

        ######################################
        # Additional attributes: add to README
        curr_object['theta1'] = theta1
        curr_object['area'] = area
        # curr_object['rho'] = rho
        ######################################

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


"""
"""


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
        object_label = item['object_label']
        x_position = item['x_position']
        y_position = item['y_position']
        min_moment = item['min_moment'] # might matter?
        orientation = item['orientation'] # might matter?
        roundness = item['roundness'] # matters?
        area = item['area']
        # rho = item["rho"]

        for model_item in database_in:
            model_object_label = model_item['object_label']
            model_x_position = model_item['x_position']
            model_y_position = model_item['y_position']
            model_min_moment = model_item['min_moment'] # might matter?
            model_orientation = model_item['orientation'] # might matter?
            model_roundness = model_item['roundness'] # matters?
            model_area = model_item['area']

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




# # cv2.imwrite('two_objects_overlaid.jpg', overlays_out)

# img = cv2.imread('many_objects_1.pgm', 0)
# binary_out = p1(img, 110)
# cv2.imwrite('many_objects_1_binary.jpg', binary_out)
# labels_out = p2(binary_out)
# cv2.imwrite('many_objects_1_labeled.jpg', labels_out)
# database_out, overlays_out = p3(labels_out)
# # cv2.imwrite('many_objects_1_overlaid.jpg', overlays_out)
        
# img = cv2.imread('many_objects_2.pgm', 0)
# binary_out = p1(img, 110)
# cv2.imwrite('many_objects_2_binary.jpg', binary_out)
# labels_out = p2(binary_out)
# cv2.imwrite('many_objects_2_labeled.jpg', labels_out)
# database_out, overlays_out = p3(labels_out)
# # cv2.imwrite('many_objects_2_overlaid.jpg', overlays_out)

