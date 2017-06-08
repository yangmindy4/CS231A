
import numpy as np
from scipy import misc
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import convolve
from sklearn.cluster import KMeans, MeanShift
import sys
import operator
import scipy.misc
from skimage.feature import corner_harris, corner_foerstner, corner_subpix, corner_peaks, peak_local_max, corner_shi_tomasi, canny
from matplotlib import pyplot as plt
import pdb
from skimage import io, color, morphology
from scipy.signal import convolve2d
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)
from skimage import data
import matplotlib.pyplot as plt
from matplotlib import cm


def createFilter(degree, size):
    #size is odd
    flip = False
    if degree > 180:
        degree -= 180
        flip = True
    center = size/2
    top = np.zeros((center+1, size))
    for i in range(size):
        for j in range(center + 1): 

            x_dist = i - center
            y_dist = center - j

            if y_dist == 0 and x_dist < 0:
                top[j, i] = 1
            else:

                if x_dist == 0:
                    slope = float('inf')
                else:
                    slope = (y_dist)/float(x_dist)
                angle = np.degrees(np.arctan(slope))
                if angle < 0:
                    angle += 180
                if angle < degree:
                    top[j, i] = 0
                else:
                    top[j, i] = 1
    bottom = np.zeros((center, size))
    bottom[np.flipud(np.fliplr(top[0:center, :])) == 0] = 1
    filt = np.concatenate((top, bottom), axis=0) 
    if flip:
        filt = 1-filt
    return filt

def intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return [None, None]

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]

# def kmeans():

def houghAverage():
    image = readImage("images/envelope2min.jpg")
    height, width = image.shape
    image = gaussian_filter(image, sigma=2)
    image[image > 0.5] = 1
    image[image <= 0.5] = 0
    edges = canny(image, sigma=3)
    lines = probabilistic_hough_line(edges, threshold=20, line_gap=2, line_length=100)
    lines = np.array(lines)
    vecs = lines[:, 1, :] - lines[:, 0, :]
    slopes = vecs[:, 1]/vecs[:, 0].astype(float)

    intersections = set()
    for i, l1 in enumerate(lines):
        for j, l2 in enumerate(lines):
            if i == j:
                continue
            intersections.add(tuple(intersection(l1,  l2)))
    intersections = np.asarray(list(intersections))
    idx = np.where((intersections[:, 1] < height) & (intersections[:, 1] >= 0) & (intersections[:, 0] < width) &(intersections[:, 0] >= 0))[0]
    intersections = intersections[idx]
    clusters = MeanShift(bandwidth=10).fit_predict(intersections)
    coords = []
    for i in range(np.max(clusters)+1):
        idx = np.where(clusters == i)[0]
        cluster_int = intersections[idx]
        coords.append(np.mean(cluster_int, axis=0))
    coords = np.array(coords)
    coords = np.hstack((np.expand_dims(coords[:, 1], 1), np.expand_dims(coords[:, 0], 1)))
    return np.array(coords)



def hough():

    ground_truth = np.array([[[321, 477],[468, 494]], 
                             [[1609, 486],[1976, 372]], 
                             [[2856, 1007],[2913, 1366]], 
                             [[2733, 1398],[2921, 1374]], 
                             [[158, 1504],[362, 1504]], 
                             [[321, 477],[232, 836]]])

    ground_truth_idx = np.array([26, 7, 8, 11, 3, 1])
    image = readImage("images/envelope2min.jpg")
    image = gaussian_filter(image, sigma=2)
    image[image > 0.5] = 1
    image[image <= 0.5] = 0
    edges = canny(image, sigma=3)
    lines = probabilistic_hough_line(edges, threshold=20, line_gap=2, line_length=100)
    ground_truth_lines = np.array(lines)[ground_truth_idx]

    # vecs = lines[:, 1, :] - lines[:, 0, :]
    # vecs = vecs[ground_truth_idx]

    intersections = []
    for i, p in enumerate(ground_truth):
        adj1 = ground_truth_lines[i-1]
        intersections.append(intersection(adj1,  ground_truth_lines[i]))
    intersections = np.array(intersections)
    coords = np.hstack((np.expand_dims(intersections[:, 1], 1), np.expand_dims(intersections[:, 0], 1)))
    # pdb.set_trace()
    # slopes = vecs[:, 1]/vecs[:, 0].astype(float)
    # num_clusters = 6
    # clusters = KMeans(n_clusters=num_clusters, n_init=20).fit_predict(np.expand_dims(np.degrees(np.arctan(slopes)), 1))
    # endpoints = []
    # for i in range(num_clusters):
    #     idx = np.where(clusters == i)[0]
    #     cluster_vecs = vecs[idx]
    #     cluster_lines = lines[idx]
    #     x_min = np.min(np.reshape(cluster_lines[:, 0, 0], -1))
    #     x_max = np.max(np.reshape(cluster_lines[:, 1, 0], -1))
    #     y_min = np.min(np.reshape(cluster_lines[:, 0, 1], -1))
    #     y_max = np.max(np.reshape(cluster_lines[:, 1, 1], -1))
    #     endpoints.append([x_min, y_min, x_max, y_max])

    # endpoints = np.array(endpoints)
    # pdb.set_trace()
    # adjacency = []
    # for i, p1 in enumerate(endpoints):
    #     adj1 = 0
    #     adj2 = 0
    #     pair_dist1 = float('inf')
    #     pair_dist2 = float('inf')
    #     for j, p2 in enumerate(endpoints):
    #         if i == j: 
    #             continue
    #         dist1 = np.linalg.norm(p1[0:2]-p2[2:4])
    #         dist2 = np.linalg.norm(p1[2:4]-p2[0:2])
    #         if dist1 < pair_dist1:
    #             pair_dist1 = dist1
    #             adj1 = j
    #         if dist2 < pair_dist2:
    #             pair_dist2 = dist2
    #             adj2 = j
    #     adjacency.append([i, adj1, adj2])

    # coords = []
    # for i, p in enumerate(endpoints):
    #     endpoints[adjacency[i][1]]
    #     # adjacency[i][2]
    #     # endpoints
    # for i, p in enumerate(endpoints):
    #     l1 = endpoints[adjacency[i][1]]
    #     l2 = endpoints[adjacency[i][2]]
    #     coords.append(np.cross(l1, l2))
        #finding adjacent line clusters


    # Generating figure 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')

    ax[1].imshow(edges, cmap=cm.gray)
    ax[1].set_title('Canny edges')

    ax[2].imshow(edges * 0)
    for line in ground_truth_lines:
        p0, p1 = line
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].set_xlim((0, image.shape[1]))
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_title('Probabilistic Hough')

    for a in ax:
        a.set_axis_off()
        a.set_adjustable('box-forced')

    plt.tight_layout()
    plt.show()
    return coords

def generateFilterImages():
    for degree in range(95, 360, 5):
            size = 81
            filt = createFilter(degree, size)
            scipy.misc.imsave("images/filt" + str(degree) + ".jpg", filt)

def suppressCorners(coords, img, max_corners):
    convolutions = []

    for i, c in enumerate(coords):
        row_convolutions = []

        for degree in range(0, 180, 5):
            size = 21
            filt = createFilter(degree, size)
            scipy.misc.imsave("filt" + str(degree), filt)
            half = size/2
            if np.reshape(img[c[0]-half:c[0]+half+1, c[1]-half:c[1]+half+1], -1).shape[0] == 0:
                row_convolutions.append(0)
                continue
            row_convolutions.append(np.convolve(np.reshape(filt, -1), np.reshape(img[c[0]-half:c[0]+half+1, c[1]-half:c[1]+half+1], -1), 'valid')[0])
        convolutions.append(row_convolutions)
    pdb.set_trace()
    threshold = (np.array(convolutions) > 150).astype(int)
    idx = np.argsort(np.sum(threshold, axis=1))[0:max_corners]
    return coords[idx]


def readImage(filename):
    img = misc.imread(filename, flatten=True)

    return img/np.max(img)

# def findCorners(img, window_size, k, thresh):
#     """
#     Finds and returns list of corners and new image with corners drawn
#     :param img: The original image
#     :param window_size: The size (side length) of the sliding window
#     :param k: Harris corner constant. Usually 0.04 - 0.06
#     :param thresh: The threshold above which a corner is counted
#     :return:
#     """
#     #Find x and y derivatives
#     pdb.set_trace()
#     dy, dx = np.gradient(img)
#     Ixx = dx**2
#     Ixy = dy*dx
#     Iyy = dy**2
#     height = img.shape[0]
#     width = img.shape[1]

#     cornerList = []
#     newImg = img.copy()
#     offset = window_size/2

#     #Loop through image and find our corners
#     print "Finding Corners..."
#     for y in range(offset, height-offset):
#         for x in range(offset, width-offset):

#             #Calculate sum of squares
#             windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
#             windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
#             windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
#             Sxx = windowIxx.sum()
#             Sxy = windowIxy.sum()
#             Syy = windowIyy.sum()

#             #Find determinant and trace, use to get corner response
#             det = (Sxx * Syy) - (Sxy**2)
#             trace = Sxx + Syy
#             r = det - k*(trace**2)

#             #If corner response is over threshold, color the point and add to corner list
#             if r > thresh:
#                 print x, y, r
#                 cornerList.append([x, y, r])
#                 newImg.itemset((y, x, 0), 0)
#                 newImg.itemset((y, x, 1), 0)
#                 newImg.itemset((y, x, 2), 255)
#     pdb.set_trace()
#     return newImg, cornerList

def main():
    """
    Main parses argument list and runs findCorners() on the image
    :return: None
    """

    # coords = houghAverage()
    generateFilterImages()
    # coords = hough()
    # img = readImage("images/envelope2min.jpg")
    # img = gaussian_filter(img, sigma=2)
    # img[img > 0.5] = 1
    # img[img <= 0.5] = 0
    # # plt.imshow(img)
    # # plt.show()
    # # coords = corner_peaks(corner_shi_tomasi(img, sigma=1), min_distance=20)
    # # coords = suppressCorners(coords, img, 12)
    # # pdb.set_trace()
    # # # coords_subpix = corner_subpix(img, coords, alpha=2, window_size=100)
    # fig, ax = plt.subplots()
    # ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    # ax.plot(coords[:, 1], coords[:, 0], '.r', markersize=10)
    # # ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
    # plt.show()

    #testing out filter
    # img = readImage("images/envelope1min.jpg")
    # # img = gaussian_filter(img, sigma=2)
    # img[img > 0.5] = 1
    # img[img <= 0.5] = 0
    # # plt.imshow(img)
    # # plt.show()
    # coords = corner_peaks(corner_harris(img))
    # coords = suppressCorners(coords, img, 4)
    # pdb.set_trace()
    # # coords_subpix = corner_subpix(img, coords, alpha=2, window_size=100)
    # fig, ax = plt.subplots()
    # ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    # ax.plot(coords[:, 1], coords[:, 0], '.b', markersize=3)
    # # ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
    # plt.show()

#     window_size = 5
#     k = 0.04
#     thresh = 10000 
#     # finalImg, cornerList = findCorners(img, int(window_size), float(k), int(thresh))


    # img = color.rgb2gray(io.imread('images/envelope1.jpg'))

    # edges1 = feature.canny(img, sigma=3)
    # fig, ax = plt.subplots()
    # ax.imshow(edges1, interpolation='nearest', cmap=plt.cm.gray)

if __name__ == "__main__":
    main()
