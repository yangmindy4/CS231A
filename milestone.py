import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import matplotlib.path as mppath
import math
import pdb
from scipy.spatial import ConvexHull
import copy

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

def triangle_example(H):
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

# Transform the portion of the image specified by indices
# according to the homography (inverse of it). Use
# bilinear interpolation
def transform_image(H, image, indices):
    H_inv = np.linalg.inv(H)
    # corresponding points in the original folded image that map to points
    # specified by indices
    corresponding_points = calculate_projection(H_inv, indices)
    new_image = copy.deepcopy(image)
    for i in range(len(indices)):
        new_image[int(corresponding_points[i][1]), int(corresponding_points[i][0])] = 0
    for i in range(len(indices)):
        new_image[indices[i][1], indices[i][0]] = image[int(corresponding_points[i][1]), int(corresponding_points[i][0])]
    return new_image


def dog_example():
    dog_image = mpimg.imread("images/Dog2.jpg")
    plt.imshow(dog_image)
    plt.show()
    #Dog1.jpg flat portion corners
    # flat_corners = np.array([
    #     [378.78, 685.616],
    #     [1394.08, 770.14],
    #     [1364.68, 1527.18],
    #     [89.466, 1450.01]])
    # folded_corners = np.array([
    #     [1394.08, 770.14],
    #     [2371.62, 303.419],
    #     [2588.77, 1042.09],
    #     [1364.68, 1527.18]
    #     ])

    #Dog2.jpg flat portion corners
    # flat_corners = np.array([
    #     [428.145, 657.36],
    #     [1460.76, 777.252],
    #     [1398.88, 1554.62],
    #     [126.481, 1423.12]])
    ground_truth = np.array([
        [0, 0],
        [1200, 0],
        [1200, 1050],
        [0, 1050]
        ])
    folded_corners = np.array([
        [1460.76, 777.252],
        [2284.54, 57.8991],
        [2385.09, 815.927],
        [1398.88, 1554.62],
        ])
    top_right_projected = 2*flat_corners[1]-flat_corners[0]
    bottom_right_projected = 2*flat_corners[2]-flat_corners[3]
    # plt.plot(top_right_projected[0], top_right_projected[1], 'bo')
    # plt.plot(bottom_right_projected[0], bottom_right_projected[1], 'bo')

    projection = np.r_[flat_corners[1][None,:], top_right_projected[None,:], \
        bottom_right_projected[None,:],flat_corners[2][None,:]]
    H = compute_homography(folded_corners, projection)
   
    reprojection = calculate_projection(H, folded_corners)

    folded = mppath.Path(np.r_[reprojection, reprojection[0][None,:]], closed=True)
    xx, yy = np.meshgrid(np.arange(0, dog_image.shape[1]), np.arange(0, dog_image.shape[0]))
    xx = np.reshape(xx, -1)
    yy = np.reshape(yy, -1)
    zipped = zip(xx, yy)
    flattened_portion = np.array(zipped)[np.where(folded.contains_points(zipped) == True)[0]]
    new_image = transform_image(H, dog_image, flattened_portion)
    plt.imshow(new_image)
    plt.show()

def main():
    # folded_envelope = mpimg.imread("images/Envelope2.jpg")
    # plt.imshow(folded_envelope)
    # #flat portion corners
    # flat_corners = np.array([
    #     [360.481, 597.411],
    #     [1519.57, 646.386],
    #     [1462.43, 1670.79],
    #     [99.277, 1605.49]])
    # folded_corners = np.array([
    #     [1519.57, 646.386],
    #     [2756.2, 262.744],
    #     [2964.35, 1499.38],
    #     [1462.43, 1670.79]
    #     ])
    # top_right_projected = 2*flat_corners[1]-flat_corners[0]
    # bottom_right_projected = 2*flat_corners[2]-flat_corners[3]
    # # plt.plot(top_right_projected[0], top_right_projected[1], 'bo')
    # # plt.plot(bottom_right_projected[0], bottom_right_projected[1], 'bo')

    # projection = np.r_[flat_corners[1][None,:], top_right_projected[None,:], \
    #     bottom_right_projected[None,:],flat_corners[2][None,:]]
    # H = compute_homography(folded_corners, projection)
   
    # reprojection = calculate_projection(H, folded_corners)

    # folded = mppath.Path(np.r_[reprojection, reprojection[0][None,:]], closed=True)
    # xx, yy = np.meshgrid(np.arange(0, folded_envelope.shape[1]), np.arange(0, folded_envelope.shape[0]))
    # xx = np.reshape(xx, -1)
    # yy = np.reshape(yy, -1)
    # zipped = zip(xx, yy)
    # flattened_portion = np.array(zipped)[np.where(folded.contains_points(zipped) == True)[0]]
    # new_image = transform_image(H, folded_envelope, flattened_portion)
    # plt.imshow(new_image)
    # plt.show()
    # triangle_example(H)
    dog_example()
   

if __name__ == '__main__':
    main()