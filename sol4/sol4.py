from sol4_utils import *
from sol4_add import non_maximum_suppression as nms, spread_out_corners as spoc, least_squares_homography as lsh
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from numpy.matlib import repmat

# TODO - ret type in all functions!

K = 0.04
BLUR_KER_SIZE = 3
DERIVE_KER = np.array([[1, 0, -1]], dtype=np.float32)
M = 7
N = 7
SPOC_RADIUS = 3
DESC_RADIUS = 3


def derive_img(im, axis=0):
    """
        Derives an image in the given axis using simple convolution with [-1, 0 ,1]

        Input:
            a grayscale images of type float32
    """
    if axis != 0:
        return derive_img(im.transpose()).transpose()
    return convolve2d(im, DERIVE_KER, mode='same')


def get_blured_mat_mul(im1, im2):
    return blur_spatial(np.multiply(im1, im2), BLUR_KER_SIZE)

# in my code i should use spread_out_corners (play with n,m but start with n=m=7)
def harris_corner_detector(im):
    """
    Basic harris corner detector (not scale invariant)
    :param im: − grayscale image to find key points inside
    :return: pos - An array with shape (N,2) of [x,y] key points locations in im.
    """
    # Get Ix and Iy with [1,0,-1]
    # blur Ix2, Iy2 and IxIy with blur_spatial kernel 3
    # For each pixel we have M: [[Ix2 IxIy],[IyIx, Iy2]]
    # find R = det(M) − k(trace(M))^2 with k=0.04
    # find response image with R for each pixel
    # use non_maximum_supression to get a binary image with the local maximum points
    # Return the xy coordinates of the corners.
    Ix, Iy, = derive_img(im, 0), derive_img(im, 1)
    Ix2, Iy2, IxIy = get_blured_mat_mul(Ix, Ix), get_blured_mat_mul(Iy, Iy), get_blured_mat_mul(Ix, Iy)
    trace_M = Ix2+Iy2
    det_M = np.multiply(Ix2,Iy2) - np.power(IxIy, 2)
    R = det_M - K*np.power(trace_M,2)
    return np.fliplr(np.array(np.where(nms(R))).transpose())


def map_coord_2_level(pos, li=0, lj=2):
    return pos * 2**(li-lj)

def get_windows_coords(pos, desc_rad):
    if pos.ndim > 1:
        coords_x = get_windows_coords(pos[:, 0], desc_rad)
        coords_y = get_windows_coords(pos[:, 1], desc_rad)
        return np.hstack((coords_x, coords_y))

    k = desc_rad * 2 + 1
    coords = repmat(pos[:, np.newaxis], 1, k**2)
    inddiff = repmat(np.arange(-desc_rad, desc_rad+1), pos.size, k)
    coords += inddiff
    return coords.reshape((1,-1)).transpose()
    # coords = np.zeros((pos.shape[0]*(k**2), 2))
    # coords[::(k+1)*desc_rad,:] = pos
    # coords[k//2,k//2,:] = 0
    # return coords
# TODO - not done!

def sample_descriptor(im, pos, desc_rad):
    """

    :param im: − grayscale image to sample within.
    :param pos: − An array with shape (N,2) of [x,y] positions to sample descriptors in im.
    :param desc_rad: − ”Radius” of descriptors to compute (see below).
    :return: desc − A 3D array with shape (K,K,N) containing the ith descriptor at desc(:,:,i).
                The per−descriptor dimensions KxK are related to the desc rad argument as follows K = 1+2∗desc rad.
    """
    k = desc_rad * 2 + 1
    pos_in_l3 = map_coord_2_level(pos)
    coords = get_windows_coords(pos_in_l3, desc_rad).transpose()
    desc = map_coordinates(im, coords).reshape((k**2, -1)).transpose()

    # normalize dsec
    desc = desc - np.mean(desc, axis=1)[:, np.newaxis]
    desc = desc / np.linalg.norm(desc, axis=1)[:, np.newaxis]
    return desc.reshape((k,k,-1)).astype(np.float32)

def find_features(pyr):
    """
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return:
        pos − An array with shape (N,2) of [x,y] feature location per row found in the (third pyramid level of the)
                image. These coordinates are provided at the pyramid level pyr[0].
        desc − A feature descriptor array with shape (K,K,N).
    """
    pos = spoc(pyr[0], M, N, SPOC_RADIUS)
    desc = sample_descriptor(pyr[2], pos, DESC_RADIUS)
    return pos, desc

# TODO - in the next two functions need to make sure the axes and indexing are correct
def get_sec_largest(mat, axis=0):
    if axis != 0:
        return get_sec_largest(mat.transpose()).transpose()
    m = mat.copy()
    m[m.argmax(axis=0)] = m.min()
    return m.max(axis=0)

def match_features(desc1, desc2, min_score):
    """
    :param desc1: A feature descriptor array with shape (K,K,N1).
    :param desc2: A feature descriptor array with shape (K,K,N2).
    :param min_score: Minimal match score between two descriptors required to be regarded as corresponding points.
    :return:
        match ind1 − Array with shape (M,) and dtype int of matching indices in desc1.
        match ind2 − Array with shape (M,) and dtype int of matching indices in desc2.
    """
    d1 = desc1.reshape((desc1.shape[2], -1))
    d2 = desc2.reshape((desc2.shape[2], -1)).transpose()
    scores = np.matmul(d1, d2)
    sec_larg_cols = get_sec_largest(scores)[np.newaxis,:]
    sec_larg_rows = get_sec_largest(scores, 1)[:,np.newaxis]
    first_prop = scores >= sec_larg_cols
    second_prop = scores >= sec_larg_rows
    third_prop = scores >= min_score
    matches = np.where(np.logical_and(np.logical_and(first_prop, second_prop), third_prop))
    return matches[0], matches[1]


def apply_homography(pos1, H12):
    """
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates in image i+1
                obtained from transforming pos1 using H12.
    """
    xyz1 = np.ones((3, pos1.shape[0]))
    xyz1[0:2,:] = pos1.transpose()
    xyz2 = np.matmul(H12, xyz1)
    xy2 = xyz2[0:2,:] / xyz2[2,:]
    return xy2.transpose()


def ransac_homography(pos1, pos2, num_iters, inlier_tol):
    """
    :param pos1: An Array, with shape (N,2) containing n rows of [x,y] coordinates of matched points.
    :param pos2: see pos1
    :param num_iters: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :return:
            H12 − A 3x3 normalized homography matrix.
            inliers − An Array with shape (S,) where S is the number of inliers, containing the indices in
                    pos1/pos2 of the maximal set of inlier matches found.
    """
    N = range(pos1.shape[0])
    inliers = np.array([])
    for n in range(num_iters):
        rnd_ind = np.random.choice(N, 4)
        H = lsh(pos1[rnd_ind,:], pos2[rnd_ind, :])
        if H is None: continue
        sqdiff = np.linalg.norm(apply_homography(pos1, H) - pos2, axis=1)
        inlierstemp = np.where(sqdiff < inlier_tol)[0]
        if inlierstemp.size > inliers.size:
            inliers = inlierstemp
    return lsh(pos1[inliers], pos2[inliers]), inliers


def display_matches(im1, im2, pos1, pos2, inliers):
    """
    :param im1: grayscale image
    :param im2: grayscale image
    :param pos1, pos2: − Two arrays with shape (N,2) each, containing N rows of [x,y] coordinates of matched
                            points in im1 and im2 (i.e. the match of the ith coordinate is pos1[i,:] in
                            im1 and pos2[i,:] in im2)
    :param inliers: An array with shape (S,) of inlier matches (e.g. see output of ransac homography)
    """
    im = np.hstack((im1, im2))
    plt.figure()
    if im.ndim == 2:
        plt.imshow(im, cmap=plt.cm.gray)
    else:
        plt.imshow(im)
    plt.hold(True)
    outliers = np.delete(np.arange(pos1.shape[0]), inliers)
    x1, y1, x2, y2 = pos1[:,0], pos1[:,1], pos2[:,0]+im1.shape[1], pos2[:,1]
    x, y = np.hstack((x1, x2)), np.hstack((y1, y2))
    plt.scatter(x,y, c='r')
    plt.plot(np.vstack((x1[outliers][np.newaxis,:], x2[outliers][np.newaxis,:])),
             np.vstack((y1[outliers][np.newaxis,:], y2[outliers][np.newaxis,:])), c ='b', lw = .4,
             ms = 0)
    plt.plot(np.vstack((x1[inliers][np.newaxis, :], x2[inliers][np.newaxis, :])),
             np.vstack((y1[inliers][np.newaxis, :], y2[inliers][np.newaxis, :])), c='y', lw=.4,
             ms=0)

    #plt.axes('off')
    plt.show()

