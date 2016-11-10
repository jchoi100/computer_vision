
# f = open('test_out.txt', 'w')
# for x in range(3):
#     for y in range(3):
#         for z in range(3):
#             f.write('(' + str(x + 0.1) + ', ' + str(y) + ', ' + str(z) + ')\n')
from math import sqrt

El = [[1, 2, 3, 4, 5], [6, 7, 8,  9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21,22,23,24,25]]
Er = [[101, 102, 103, 104, 105], [106, 107, 108, 109, 110], [111, 112, 113, 114, 115],[116, 117, 118, 119, 120], [121,122,123,124,125]] 
WINDOW_SIZE = 3

def compute_ncc(u, v, u2):
    denom1, denom2, numer = 0.0, 0.0, 0.0
    for i in range(WINDOW_SIZE):
        for j in range(WINDOW_SIZE):
            denom1 += El[v + j][u + i]**2
            denom2 += Er[v + j][u2 + i]**2
            numer += (El[v + j][u + i] - Er[v + j][u2 + i])**2
    return numer / sqrt(denom1 * denom2)

num_rows, num_cols = len(El), len(El[0])
for v in range(num_rows - WINDOW_SIZE):
    for u in range(num_cols - WINDOW_SIZE):
        # v is fixed even in Er
        for u2 in range(num_cols - WINDOW_SIZE):
            print(compute_ncc(u, v, u2))
