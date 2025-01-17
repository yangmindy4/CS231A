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
# TODO: bilinear interpolation
def transform_image(H, orig_image, new_image, indices):
    H_inv = np.linalg.inv(H)
    # corresponding points in the original folded image that map to points
    # specified by indices
    corresponding_points = calculate_projection(H_inv, indices)
    # for i in range(len(indices)):
    #     new_image[int(corresponding_points[i][1]), int(corresponding_points[i][0])] = 0
    for i in range(len(indices)):
        new_image[indices[i][1], indices[i][0]] = orig_image[int(corresponding_points[i][1]), int(corresponding_points[i][0])]

    return new_image

def dog_example():
    dog_image = mpimg.imread("images/Dog2.jpg")
    # plt.imshow(dog_image)
    # plt.show()
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
    flat_corners = np.array([
        [428.145, 657.36],
        [1460.76, 777.252],
        [1398.88, 1554.62],
        [126.481, 1423.12]])
    flat_ground_truth = np.array([
        [0, 0],
        [1199, 0],
        [1199, 1049],
        [0, 1049]
        ])
    folded_corners = np.array([
        [1460.76, 777.252],
        [2284.54, 57.8991],
        [2385.09, 815.927],
        [1398.88, 1554.62],
        ])
    folded_ground_truth = np.array([
        [1199, 0],
        [2399, 0],
        [2399, 1049],
        [1199, 1049]
        ])

    H_flat = compute_homography(flat_corners, flat_ground_truth)
    H_folded = compute_homography(folded_corners, folded_ground_truth)
   
    new_image = np.zeros((1050, 2400, 3),dtype='uint8')
    folded = mppath.Path(np.r_[folded_ground_truth, folded_ground_truth[0][None,:]], closed=True)
    xx, yy = np.meshgrid(np.arange(0, new_image.shape[1]), np.arange(0, new_image.shape[0]))
    xx = np.reshape(xx, -1)
    yy = np.reshape(yy, -1)
    zipped = zip(xx, yy)
    folded_portion = np.array(zipped)[np.where(folded.contains_points(zipped) == True)[0]]
    transform_image(H_folded, dog_image, new_image, folded_portion)

    flattened = mppath.Path(np.r_[flat_ground_truth, flat_ground_truth[0][None,:]], closed=True)
    xx, yy = np.meshgrid(np.arange(0, new_image.shape[1]), np.arange(0, new_image.shape[0]))
    xx = np.reshape(xx, -1)
    yy = np.reshape(yy, -1)
    zipped = zip(xx, yy)
    flattened_portion = np.array(zipped)[np.where(flattened.contains_points(zipped) == True)[0]]
    transform_image(H_flat, dog_image, new_image, flattened_portion)
    plt.imshow(new_image)
    plt.show()

def two_folds_example():
    image = mpimg.imread("images/TwoFolds6.jpg")

    # top left corners
    folded_tl = np.array([
        [784.506, 376.0],
        [1698.97, 532.875],
        [1610.97, 1137.42],
        [604.673, 995.848]])
    # top right corners
    folded_tr = np.array([
        [1698.97, 532.875],
        [2575.18, 134.948],
        [2678.48, 926.976],
        [1610.97, 1137.42]
        ])
    # bottom left corners
    folded_bl = np.array([
        [604.673, 995.848],
        [1610.97, 1137.42],
        [1461.75, 1780.22],
        [264.14, 1619.52]
        ])
    # bottom right corners
    folded_br = np.array([
        [1610.97, 1137.42],
        [2678.48, 926.976],
        [2556.05, 1263.68],
        [1461.75, 1780.22]
        ])

    ground_truth_tl = np.array([
        [0, 0],
        [1199, 0],
        [1199, 1049],
        [0, 1049]
        ])
    ground_truth_tr = np.array([
        [1199, 0],
        [2399, 0],
        [2399, 1049],
        [1199, 1049]
        ])
    ground_truth_bl = np.array([
        [0, 1050],
        [1199, 1050],
        [1199, 2099],
        [0, 2099]
        ])
    ground_truth_br = np.array([
        [1199, 1050],
        [2399, 1050],
        [2399, 2099],
        [1199, 2099]
        ])
    H_tl = compute_homography(folded_tl, ground_truth_tl)
    H_tr = compute_homography(folded_tr, ground_truth_tr)
    H_bl = compute_homography(folded_bl, ground_truth_bl)
    H_br = compute_homography(folded_br, ground_truth_br)
   
    new_image = np.zeros((2100, 2400, 3),dtype='uint8')
    tl = mppath.Path(np.r_[ground_truth_tl, ground_truth_tl[0][None,:]], closed=True)
    xx, yy = np.meshgrid(np.arange(0, new_image.shape[1]), np.arange(0, new_image.shape[0]))
    xx = np.reshape(xx, -1)
    yy = np.reshape(yy, -1)
    zipped = zip(xx, yy)
    tl_portion = np.array(zipped)[np.where(tl.contains_points(zipped) == True)[0]]
    transform_image(H_tl, image, new_image, tl_portion)

    tr = mppath.Path(np.r_[ground_truth_tr, ground_truth_tr[0][None,:]], closed=True)
    xx, yy = np.meshgrid(np.arange(0, new_image.shape[1]), np.arange(0, new_image.shape[0]))
    xx = np.reshape(xx, -1)
    yy = np.reshape(yy, -1)
    zipped = zip(xx, yy)
    tr_portion = np.array(zipped)[np.where(tr.contains_points(zipped) == True)[0]]
    transform_image(H_tr, image, new_image, tr_portion)

    bl = mppath.Path(np.r_[ground_truth_bl, ground_truth_bl[0][None,:]], closed=True)
    xx, yy = np.meshgrid(np.arange(0, new_image.shape[1]), np.arange(0, new_image.shape[0]))
    xx = np.reshape(xx, -1)
    yy = np.reshape(yy, -1)
    zipped = zip(xx, yy)
    bl_portion = np.array(zipped)[np.where(bl.contains_points(zipped) == True)[0]]
    transform_image(H_bl, image, new_image, bl_portion)

    br = mppath.Path(np.r_[ground_truth_br, ground_truth_br[0][None,:]], closed=True)
    xx, yy = np.meshgrid(np.arange(0, new_image.shape[1]), np.arange(0, new_image.shape[0]))
    xx = np.reshape(xx, -1)
    yy = np.reshape(yy, -1)
    zipped = zip(xx, yy)
    br_portion = np.array(zipped)[np.where(br.contains_points(zipped) == True)[0]]
    transform_image(H_br, image, new_image, br_portion)

    plt.imshow(new_image)
    plt.show()

def checkerboard_example():
    checkerboard_image = mpimg.imread("images/Checkerboard1.jpg")
    # plt.imshow(checkerboard_image)
    # plt.show()
    flat_corners = np.array([
        [638.507, 927.669],
        [1108.66, 836.672],
        [2223.37, 1557.06],
        [877.374, 1932.43]])
    flat_ground_truth = np.array([
        [0, 0],
        [719, 0],
        [1680, 1854],
        [0, 1854]
        ])
    folded_corners = np.array([
        [1108.66, 836.672],
        [1874.55, 70.781],
        [2720.06, 1246.16],
        [2223.37, 1557.06],
        ])
    folded_ground_truth = np.array([
        [720, 0],
        [2399, 0],
        [2399, 1854],
        [1680, 1854]
        ])
   
    H_flat = compute_homography(flat_corners, flat_ground_truth)
    H_folded = compute_homography(folded_corners, folded_ground_truth)
   

    new_image = np.zeros((1855, 2400, 3),dtype='uint8')
    folded = mppath.Path(np.r_[folded_ground_truth, folded_ground_truth[0][None,:]], closed=True)
    xx, yy = np.meshgrid(np.arange(0, new_image.shape[1]), np.arange(0, new_image.shape[0]))
    xx = np.reshape(xx, -1)
    yy = np.reshape(yy, -1)
    zipped = zip(xx, yy)
    folded_portion = np.array(zipped)[np.where(folded.contains_points(zipped) == True)[0]]
    transform_image(H_folded, checkerboard_image, new_image, folded_portion)

    flattened = mppath.Path(np.r_[flat_ground_truth, flat_ground_truth[0][None,:]], closed=True)
    xx, yy = np.meshgrid(np.arange(0, new_image.shape[1]), np.arange(0, new_image.shape[0]))
    xx = np.reshape(xx, -1)
    yy = np.reshape(yy, -1)
    zipped = zip(xx, yy)
    flattened_portion = np.array(zipped)[np.where(flattened.contains_points(zipped) == True)[0]]
    transform_image(H_flat, checkerboard_image, new_image, flattened_portion)
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
    # dog_example()
    # two_folds_example()
    checkerboard_example()
   

if __name__ == '__main__':
    main()