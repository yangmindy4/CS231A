import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import math
import pdb
from scipy.spatial import ConvexHull

def calculate_projection(H, points):
    points_homogenized = np.c_[points, np.ones(len(points))].T
    projected_points = H.dot(points_homogenized)
    projected_points /= projected_points[2]
    return projected_points[:2].T

def compute_homography(image1, image2):
    x11, x21, x31, x41 = image1[:,0]
    x12, x22, x32, x42 = image2[:,0]
    y11, y21, y31, y41 = image1[:,1]
    y12, y22, y32, y42 = image2[:,1]
    A = np.array([
        [-x11, -y11, -1, 0, 0, 0, x11*x12, y11*x12, x12],
        [0, 0, 0, -x11, -y11, -1, x11*y12, y11*y12, y12],
        [-x21, -y21, -1, 0, 0, 0, x21*x22, y21*x22, x22],
        [0, 0, 0, -x21, -y21, -1, x21*y22, y21*y22, y22],
        [-x31, -y31, -1, 0, 0, 0, x31*x32, y31*x32, x32],
        [0, 0, 0, -x31, -y31, -1, x31*y32, y31*y32, y32],
        [-x41, -y41, -1, 0, 0, 0, x41*x42, y41*x42, x42],
        [0, 0, 0, -x41, -y41, -1, x41*y42, y41*y42, y42]
    ])
    u, s, v = np.linalg.svd(A)
    solution = v[-1]
    H = solution.reshape((3,3))
    return H

def main():
    folded_envelope = mpimg.imread("images/Envelope2.jpg")
    plt.imshow(folded_envelope)
    #flat portion corners
    flat_corners = np.array([
        [360.481, 597.411],
        [1519.57, 646.386],
        [99.277, 1605.49],
        [1462.43, 1670.79]])
    folded_corners = np.array([
        [1519.57, 646.386],
        [2756.2, 262.744],
        [1462.43, 1670.79],
        [2964.35, 1499.38]
        ])
    # plt.plot(flat_corners[:,0], flat_corners[:,1], 'ro')
    top_right_projected = 2*flat_corners[1]-flat_corners[0]
    bottom_right_projected = 2*flat_corners[3]-flat_corners[2]
    # plt.plot(top_right_projected[0], top_right_projected[1], 'bo')
    # plt.plot(bottom_right_projected[0], bottom_right_projected[1], 'bo')

    projection = np.r_[flat_corners[1][None,:], top_right_projected[None,:], flat_corners[3][None,:], bottom_right_projected[None,:]]
    H = compute_homography(folded_corners, projection)
    # reprojection = calculate_projection(H, folded_corners)
    # plt.plot(reprojection[:,0], reprojection[:,1], 'ro') 
    triangle = np.array([
        [1897.69, 712.403],
        [1764.29, 1039.89],
        [2116.02, 845.823]
        ])
    projected_triangle = calculate_projection(H, triangle)
    plt.plot(triangle[0,0], triangle[0,1], 'ro')
    plt.plot(triangle[1,0], triangle[1,1], 'ro')
    plt.plot(triangle[2,0], triangle[2,1], 'ro')
    plt.plot(projected_triangle[0,0], projected_triangle[0,1], 'go')
    plt.plot(projected_triangle[1,0], projected_triangle[1,1], 'go')
    plt.plot(projected_triangle[2,0], projected_triangle[2,1], 'go')
    plt.show()
    pdb.set_trace()
    folded_area = ConvexHull(folded_corners)


if __name__ == '__main__':
    main()