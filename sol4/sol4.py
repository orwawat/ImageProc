from sol4_utils import *
from sol4_add import non_maximum_suppression as nms, spread_out_corners as spoc, least_squares_homography as lsh
import numpy as np
from scipy.ndimage.filters import convolve
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from numpy.matlib import repmat

# TODO - ret type in all functions!

K = 0.04
BLUR_KER_SIZE = 3
DERIVE_KER = np.array([[1, 0, -1]], dtype=np.float32)
M = 7
N = 7
SPOC_RADIUS = 12
DESC_RADIUS = 3


def derive_img(im, axis=0):
    """
        Derives an image in the given axis using simple convolution with [-1, 0 ,1]

        Input:
            a grayscale images of type float32
    """
    if axis != 0:
        return derive_img(im.transpose()).transpose()
    return convolve(im, DERIVE_KER, mode='reflect').astype(np.float32)


def get_blured_mat_mul(im1, im2):
    """
    Multiplies pointwise the two matrices, blur them with kernel of size 3 and return the result
    :param im1, im2: two np arrays of the same shape
    :return: blured version of im1*im2
    """
    return blur_spatial(np.multiply(im1, im2), BLUR_KER_SIZE)

# in my code i should use spread_out_corners (play with n,m but start with n=m=7)
def harris_corner_detector(im):
    """
    Basic harris corner detector (not scale invariant)
    :param im: − grayscale image to find key points inside
    :return: pos - An array with shape (N,2) of [x,y] key points locations in im.
    """
    Ix, Iy, = derive_img(im, 0), derive_img(im, 1)
    Ix2, Iy2, IxIy = get_blured_mat_mul(Ix, Ix), get_blured_mat_mul(Iy, Iy), get_blured_mat_mul(Ix, Iy)
    trace_M = Ix2+Iy2
    det_M = np.multiply(Ix2,Iy2) - np.power(IxIy, 2)
    R = det_M - K*np.power(trace_M,2)

    return np.fliplr(np.array(np.where(nms(R))).transpose()).astype(np.float32)


def map_coord_2_level(pos, li=0, lj=2):
    """
    Transforms the list of points in level li to their location in level lj in the gaussian pyramid
    :param pos: no array of points [x,y]
    :param li: level of the points
    :param lj: wanted level for the points
    :return: The mapped points in the lj level of the pyramid
    """
    return pos * 2**(li-lj)


def get_windows_coords(pos, desc_rad, axis=0):
    """
    for a given set of points (array of shpae (N,2)), creates a window of desc_rad around each
    (range of [x-rad,x+rad]X[y-rad,y+rad]), finally, reshape them to be new list of size (N*(2*rad+1)^2, 2) of points
    s.t every consecutive (2*rad+1)^2 points are a window for the corresponding original point
    :param pos: A list of points
    :param desc_rad: the radius around each point for the window
    :param axis: if pos is (n,1) (only x or y axis), uses the axis argument to generate the needed range around the
        points
    :return: list of size (N*(2*rad+1)^2, 2) of points
                s.t every consecutive (2*rad+1)^2 points are a window for the corresponding original point
    """
    if pos.ndim > 1:
        coords_x = get_windows_coords(pos[:, 0], desc_rad, axis=1)
        coords_y = get_windows_coords(pos[:, 1], desc_rad, axis=0)
        return np.hstack((coords_x, coords_y)).transpose()

    k = desc_rad * 2 + 1
    coords = repmat(pos[:, np.newaxis], 1, k**2)
    if axis == 0:
        inddiff = repmat(np.arange(-desc_rad, desc_rad+1), pos.size, k)
    else:
        inddiff = repmat(np.hstack([[i]*k for i in range(-desc_rad,desc_rad+1)]), pos.size, 1)
    coords += inddiff
    return coords.reshape((1,-1)).transpose()


def sample_descriptor(im, pos, desc_rad):
    """

    :param im: − grayscale image to sample within.
    :param pos: − An array with shape (N,2) of [x,y] positions to sample descriptors in im.
    :param desc_rad: − ”Radius” of descriptors to compute (see below).
    :return: desc − A 3D array with shape (K,K,N) containing the ith descriptor at desc(:,:,i).
                The per−descriptor dimensions KxK are related to the desc rad argument as follows K = 1+2∗desc rad.
    """
    k = desc_rad * 2 + 1
    coords = get_windows_coords(pos, desc_rad)
    desc = map_coordinates(im, coords, order=1, prefilter=False).reshape((-1, k**2))

    # normalize dsec - need to ignore wrong features (all from a smooth and constant area)
    desc = desc - np.mean(desc, axis=1)[:, np.newaxis]
    # if np.count_nonzero(desc) == 0 it is a bad feature - ignore:
    norms = np.linalg.norm(desc, axis=1)
    ignores = np.where(norms == 0)[0]
    norms[ignores] = 1
    desc = desc / norms[:, np.newaxis]
    desc[ignores,:] = 0

    return desc.reshape((-1,k,k), order='C').transpose(1,2,0).astype(np.float32)


def find_features(pyr):
    """
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return:
        pos − An array with shape (N,2) of [x,y] feature location per row found in the (third pyramid level of the)
                image. These coordinates are provided at the pyramid level pyr[0].
        desc − A feature descriptor array with shape (K,K,N).
    """
    pos = spoc(pyr[0], M, N, SPOC_RADIUS)
    pos_flipped = np.fliplr(pos)
    pos_in_l3 = map_coord_2_level(pos_flipped)
    desc = sample_descriptor(pyr[2], pos_in_l3, DESC_RADIUS)

    # remove false descriptors (constant)
    k = desc.shape[0]
    mask = np.where(np.sum(np.sum((0 == desc).astype(np.uint8), axis=0), axis=0) == k**2)[0]
    desc = np.delete(desc, mask, axis=2)
    pos = np.delete(pos, mask, axis=0)

    return pos, desc


def get_sec_largest(mat, axis=0):
    """
    A util function which returns the second largest element in each row/col (depending on the axis)
    :param mat: np array
    :param axis: along which axis to find the sec largest elemnt (same as in np.max())
    :return: same as np.max, only the second largest
    """
    if axis != 0:
        return get_sec_largest(mat.transpose()).transpose()
    m = mat.copy()
    m[(m.argmax(axis=0)[np.newaxis,:], np.arange(m.shape[1])[np.newaxis,:])] = m.min()
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
    d1 = desc1.transpose((2,0,1)).reshape((desc1.shape[2], -1))
    d2 = desc2.transpose((2,0,1)).reshape((desc2.shape[2], -1)).transpose()

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
    xyz2[2, np.where(xyz2[2,:]==0)] = 1E-10
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
        sqdiff = np.power(np.linalg.norm(apply_homography(pos1, H) - pos2, axis=1),2)
        inlierstemp = np.where(sqdiff < inlier_tol)[0]
        if inlierstemp.size > inliers.size:
            inliers = inlierstemp
    return lsh(pos1[inliers,:], pos2[inliers,:]), inliers


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

    # plt.show() TODO - need or not?



def accumulate_homographies(H_successive, m):
    """
    :param H_successive: A list of M−1 3x3 homography matrices where H successive[i] is a homography that
            transforms points from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system we would like to accumulate the given homographies towards.
    :return: H2m − A list of M 3x3 homography matrices, where H2m[i] transforms points from coordinate system i
                    to coordinate system m
    """
    H2m = [np.zeros((3,3))]*(len(H_successive)+1)
    H2m[m] = np.eye(3)
    for i in range(m-1,-1,-1):
        H2m[i] = np.matmul(H2m[i+1], H_successive[i])
        H2m[i] /= H2m[i][2, 2]
    for i in range(m+1, len(H_successive)+1):
        H2m[i] = np.matmul(H2m[i-1], np.linalg.inv(H_successive[i-1]))
        H2m[i] /= H2m[i][2, 2]
    return H2m


def extract_corners_and_center(im):
    """
    TODO
    :param im:
    :return:
    """
    return np.array([[0,0], [im.shape[1]-1,0], [im.shape[1]-1,im.shape[0]-1], [0,im.shape[0]-1],
                     [(im.shape[1]-1)//2,(im.shape[0]-1)//2]])


def get_pan_size_and_borders(ims, Hs):
    """
    :param ims: A list of grayscale images. (Python list)
    :param Hs: A list of 3x3 homography matrices. Hs[i] is a homography that transforms points from the
                coordinate system of ims [i] to the coordinate system of the panorama. (Python list)
    :return: sz − shape as a tuple (rows,cols) of the final panorama image
    """
    minx, maxx, miny, maxy = 0,0,0,0
    borders = [0] * (len(ims)+1)
    last_center_x = None
    for i in range(len(ims)):
        warped_corners = apply_homography(extract_corners_and_center(ims[i]), Hs[i])
        minx = min(minx, warped_corners[:,0].min())
        maxx = max(maxx, warped_corners[:,0].max())
        miny = min(miny, warped_corners[:,1].min())
        maxy = max(maxy, warped_corners[:,1].max())
        if last_center_x is not None:
            borders[i] = int((last_center_x + warped_corners[-1,0]) // 2)
        last_center_x = warped_corners[-1,0]
    borders = (np.asarray(borders) - minx).astype(np.int)
    borders[0] = 0
    borders[-1] = int(maxx-minx+1)
    height, width = int(maxy-miny+1), int(maxx-minx+1)
    x, y = np.meshgrid(np.linspace(minx, maxx, width), np.linspace(miny, maxy, height))
    return (height, width), borders, x,y


def back_warp(im, H, x, y):
    """
    TODO
    :param im:
    :param H:
    :param x:
    :param y:
    :return:
    """
    warped_im_coords = np.hstack((x.reshape((-1, 1)), y.reshape((-1, 1))))
    warped_im_coords = apply_homography(warped_im_coords, np.linalg.inv(H))
    warped_im = map_coordinates(im, np.flipud(warped_im_coords.T), order=1, prefilter=False)
    return warped_im.reshape(x.shape)


def merge_panorama(panorama, temp_panorama, mask, levels):
    """
    TODO
    :param panorama:
    :param temp_panorama:
    :param mask:
    :param levels:
    :return:
    """
    return pyramid_blending(panorama, temp_panorama, mask, levels, 5, 5)


def render_panorama(ims, Hs):
    """
    :param ims: A list of grayscale images. (Python list)
    :param Hs: A list of 3x3 homography matrices. Hs[i] is a homography that transforms points from the
                coordinate system of ims [i] to the coordinate system of the panorama. (Python list)
    :return: panorama − A grayscale panorama image composed of vertical strips, backwarped using homographies
                    from Hs, one from every image in ims.
    """
    levels = 6
    pow2lv = 2**(levels-1)
    sz, borders, x,y = get_pan_size_and_borders(ims, Hs)
    origsz = sz
    sz = (sz[0] if sz[0] % pow2lv == 0 else sz[0] + pow2lv - sz[0] % pow2lv,
          sz[1] if sz[1] % pow2lv == 0 else sz[1] + pow2lv - sz[1] % pow2lv)
    panorama = np.zeros(sz)
    temp_panorama = np.zeros(sz)
    mask = np.zeros(sz, dtype=bool)

    for i in range(len(ims)):
        temp_panorama[:] = 0
        mask[:] = False
        bstart, bend = borders[i], borders[i + 1]
        mask[:, :bstart] = True
        if i == 0:
            panorama[:origsz[0], :origsz[1]] = back_warp(ims[i], Hs[i], x, y)
        else:
            temp_panorama[:origsz[0], :origsz[1]] = back_warp(ims[i], Hs[i], x, y)
            find_best_slice(panorama[:, borders[i-1]:borders[i+1]], temp_panorama[:, borders[i-1]:borders[i+1]])
            panorama = merge_panorama(panorama, temp_panorama, mask, levels)

    return panorama[:origsz[0], :origsz[1]]

def max_y(im):
    """
    TODO
    :param im:
    :return:
    """
    return np.where(np.sum(im, axis=1) != 0)[0][-1]

def find_best_slice(im1, im2):
    """
    TODO
    take only the sub parts that include only the overlapping parts
    where it equals 0 in the left image, take it from the right?
    :param im1:
    :param im2:
    :return:
    """
    maxy = max_y(im1)
    from scipy.ndimage.filters import minimum_filter
    E = np.power(im1-im2, 2)
    plt.figure()
    plt.imshow(E, cmap=plt.cm.gray)
    for r in range(1,maxy):
        rel_min = minimum_filter(E[r-1, :], size=1)
        E[r, :] += rel_min[1]
    path = np.zeros(maxy, dtype=np.int)
    path[-1] = np.argmin(E[maxy, :])
    for r in range(maxy-2, -1, -1):
        if path[r+1] == 0:
            s, i = 0, 0
        else:
            s, i = int(path[r+1]-1), 1
        e = int(min(path[r+1]+2, E.shape[1]+1))
        path[r] = path[r+1] + np.argmin(E[r, s:e]) - i

    res = np.zeros(im1.shape)
    for i in range(path.size):
        E[i,path[i]]=1
        res[i,:path[i]+1]=1
    plt.figure()
    plt.imshow(res, cmap=plt.cm.gray)
    res = np.multiply(res, im1) + np.multiply(1-res, im2)
    plt.figure()
    plt.imshow(res, cmap=plt.cm.gray)
    plt.show(block=True)

    return path
