
import numpy as np
from scipy import misc
import sys
import operator
from skimage.feature import corner_harris, corner_foerstner, corner_subpix, corner_peaks
from matplotlib import pyplot as plt
import pdb
from skimage import io, color, morphology
from scipy.signal import convolve2d
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import feature

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
                if degree > 90.0:
                    degree -= 180.0
                if angle < degree:
                    top[j, i] = 0
                else:
                    top[j, i] = 1
    bottom = np.zeros((center, size))
    bottom[np.flipud(np.fliplr(top[0:center, :])) == 0] = 1
    filt = np.concatenate((top, bottom), axis=0) 
    pdb.set_trace()
    if flip:
        filt = 1-filt
    return filt

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
    img = readImage("images/envelope1min.jpg")
    # img[img > 0.5] = 1
    # img[img <= 0.5] = 0
    # coords = corner_peaks(corner_harris(img))
    # coords_subpix = corner_subpix(img, coords, window_size=13)
    # fig, ax = plt.subplots()
    # ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    # ax.plot(coords[:, 1], coords[:, 0], '.b', markersize=3)
    # ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
    # plt.show()
#     window_size = 5
#     k = 0.04
#     thresh = 10000 
#     # finalImg, cornerList = findCorners(img, int(window_size), float(k), int(thresh))


    # img = color.rgb2gray(io.imread('images/envelope1.jpg'))

    # edges1 = feature.canny(img, sigma=3)
    # fig, ax = plt.subplots()
    # ax.imshow(edges1, interpolation='nearest', cmap=plt.cm.gray)
    # plt.show()
    createFilter(210, 7)

if __name__ == "__main__":
    main()
