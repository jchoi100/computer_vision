import numpy as np
import cv2

def p2(binary_in): # return labels_out
    """
    Sequential labeling algorithm that segments
    a binary image into several connected regions.
    """
    label_count = 1
    s = (len(binary_in), len(binary_in[0]))
    labels_out = np.zeros(s)
    total = len(binary_in) * len(binary_in[0])
    equivalence_relationships = UnionFind()
    for i in range(0, len(binary_in)):
        for j in range(0, len(binary_in[0])):
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
    start_label = 70
    count = 1
    offset = 30
    for i in range(len(labels_out)):
        for j in range(len(labels_out[0])):
            label = 0
            curr = labels_out[i][j]
            curr_set = equivalence_relationships[curr]
            if curr == 0:
                labels_out[i][j] = 0
            else:
                if not new_label_dict.has_key(curr_set):
                    count += 1
                    new_label_dict[curr_set] = start_label + count * offset
                labels_out[i][j] = new_label_dict[curr_set]
    return labels_out


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
