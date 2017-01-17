from sol4_utils import *
import sol4_add
import numpy as np
from scipy.ndimage.filters import convolve
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from numpy.matlib import repmat
from scipy.ndimage.filters import minimum_filter

# Constants
K = 0.04
BLUR_KER_SIZE = 3
DERIVE_KER = np.array([[1, 0, -1]], dtype=np.float32)
M = 8
N = 8
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

    return np.fliplr(np.array(np.where(sol4_add.non_maximum_suppression(R))).transpose()).astype(np.float32)


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
    Takes in an image and an array of points, and sample a secriptor for each of the points
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
    Takes in a gaussian pyramid of an image with 3 levels, find intersting features and return their location,
    as well as their corresponding descriptors
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return:
        pos − An array with shape (N,2) of [x,y] feature location per row found in the (third pyramid level of the)
                image. These coordinates are provided at the pyramid level pyr[0].
        desc − A feature descriptor array with shape (K,K,N).
    """
    pos = sol4_add.spread_out_corners(pyr[0], M, N, SPOC_RADIUS)
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
    Takes in an homography matrix and an array of points, and return the transformed points (for each points, make it
    homogeneous, transform it, and get back to 2d points)
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates in image i+1
                obtained from transforming pos1 using H12.
    """
    EPSILON = 1E-10
    xyz1 = np.ones((3, pos1.shape[0]))
    xyz1[0:2,:] = pos1.transpose()
    xyz2 = np.matmul(H12, xyz1)
    xyz2[2, np.where(xyz2[2,:]==0)] = EPSILON  # avoid division by 0
    xy2 = xyz2[0:2,:] / xyz2[2,:]
    return xy2.transpose()


def ransac_homography(pos1, pos2, num_iters, inlier_tol):
    """
    Takes in two arrays of points, and perform a ransac proccess inorder to find the best homography matrix which
    transforms the most points in pos1 to their corrsponding points in pos2 (successful transformation is
    counted by inlier tol)
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
        H = sol4_add.least_squares_homography(pos1[rnd_ind,:], pos2[rnd_ind, :])
        if H is None: continue
        sqdiff = np.power(np.linalg.norm(apply_homography(pos1, H) - pos2, axis=1), 2)
        inlierstemp = np.where(sqdiff < inlier_tol)[0]
        if inlierstemp.size > inliers.size:
            inliers = inlierstemp

    return sol4_add.least_squares_homography(pos1[inliers, :], pos2[inliers, :]), inliers


def display_matches(im1, im2, pos1, pos2, inliers):
    """
    Takes in two images, their interesting features location (corresponding arrays) and indices array marking inliers,
    and present the image side by side, with red dots for each feature, blue line connect a matched couple which is
    considered to be an outlier, and a yellow line marks inlier match
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


def accumulate_homographies(H_successive, m):
    """
    Transform a list of homography matrices from frame i to frame i+1, to a new list of homography matrices
    (longer by 1) in which every matrix i is the homography between frame i to frame m.
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
    A util function which recieves an image as an input, and a return an arrat with it's corners, and its center
    :param im: np array
    :return: a [5,2] array of [x,y] points ordered [upper-left, upper-right, bottom-right, bottom-left, center]
    """
    return np.array([[0,0], [im.shape[1]-1,0], [im.shape[1]-1,im.shape[0]-1], [0,im.shape[0]-1],
                     [(im.shape[1]-1)//2,(im.shape[0]-1)//2]])


def get_pan_size_and_borders(ims, Hs):
    """
    For a given list of images and their transformation matrices to a shared plane (the panorama plane),
    calculate the resulting panorama exact dimensions. In addition, return the division of bordered between frame, a
    corresponding mesh grady of x and y for the panorama, and the location of each corner point in each frame in the
    panorama plane (see extract_corners_and_center)
    :param ims: A list of grayscale images. (Python list)
    :param Hs: A list of 3x3 homography matrices. Hs[i] is a homography that transforms points from the
                coordinate system of ims [i] to the coordinate system of the panorama. (Python list)
    :return:
        sz − shape as a tuple (rows,cols) of the final panorama image
        x,y - the result of np meshgrid, sized like sz, and span the entire range which any point in any of the
                images has transformed onto
        warped_corners - np array of size (5,2,k) where k is the number of frames given, each channel contain
                            the result of applying corresponding homography n the results of extract_corners_and_center
    """
    # inits
    borders = [0] * (len(ims)+1)
    last_center_x = None
    warped_corners = np.zeros((5,2,len(ims)))

    # calculate warped_corners and borders between frames
    for i in range(len(ims)):
        warped_corners[:,:,i] = apply_homography(extract_corners_and_center(ims[i]), Hs[i])
        if last_center_x is not None:
            borders[i] = int((last_center_x + warped_corners[-1, 0,i]) // 2)
        last_center_x = warped_corners[-1, 0,i]

    # find corners of panorama
    minx = warped_corners[:,0,:].min()
    maxx = warped_corners[:,0,:].max()
    miny = warped_corners[:,1,:].min()
    maxy = warped_corners[:,1,:].max()

    # update borders and calculate panorama size and mesh grid
    borders = (np.asarray(borders) - minx).astype(np.int)
    borders[0] = 0
    borders[-1] = int(maxx-minx+1)
    height, width = int(maxy-miny+1), int(maxx-minx+1)
    x, y = np.meshgrid(np.linspace(minx, maxx, width), np.linspace(miny, maxy, height))

    return (height, width), borders, x, y, warped_corners


def back_warp(im, H, x, y):
    """
    For the given image and its transformation matrix to the panorama plane, and coordinate of x and y needed to be
        filled inside the panorama, perform backfilling
    :param im: A greyscale image
    :param H: Transformation matrix [3X3] from the image plane to the panorama plane
    :param x: The x coordinates to be filled (2d array)
    :param y: The y coordinates to be filled (2d array), size matching the x
    :return: a greyscale image, with size equal to x and y, backfilled from the given image
    """
    warped_im_coords = np.hstack((x.reshape((-1, 1)), y.reshape((-1, 1))))
    warped_im_coords = apply_homography(warped_im_coords, np.linalg.inv(H))
    warped_im = map_coordinates(im, np.flipud(warped_im_coords.T), order=1, prefilter=False)
    return warped_im.reshape(x.shape).astype(np.float32)


def render_panorama(ims, Hs):
    """
    From a successive list of frames, and their transformation for the same panorama plane, renders a merged panorma
    image. Uses the mean between two corresponding warped centers as the borders between warped frames, and performs
    pyramid belnding in order to seamlessly merge the images
    :param ims: A list of grayscale images. (Python list)
    :param Hs: A list of 3x3 homography matrices. Hs[i] is a homography that transforms points from the
                coordinate system of ims [i] to the coordinate system of the panorama. (Python list)
    :return: panorama − A grayscale panorama image composed of vertical strips, backwarped using homographies
                    from Hs, one from every image in ims.
    """
    # inits
    levels = 6  # Max levels for the panorama blending
    pow2lv = 2**(levels-1)
    sz, borders, x, y, warped_corners= get_pan_size_and_borders(ims, Hs)
    origsz = sz
    # find working size s.t we can perform blending on it
    sz = (sz[0] if sz[0] % pow2lv == 0 else sz[0] + pow2lv - sz[0] % pow2lv,
          sz[1] if sz[1] % pow2lv == 0 else sz[1] + pow2lv - sz[1] % pow2lv)
    panorama = np.zeros(sz)
    temp_panorama = np.zeros(sz)
    mask = np.zeros(sz, dtype=bool)

    # For each consecutive image, fill a panorama plane by backfilling from the image, then blend with current image
    for i in range(len(ims)):
        temp_panorama[:] = 0
        mask[:] = False
        bstart, bend = borders[i], borders[i + 1]
        mask[:, :bstart] = True
        if i == 0:
            # no need to blend when only 1 image has been warped
            panorama[:origsz[0], :origsz[1]] = back_warp(ims[i], Hs[i], x, y)
        else:
            temp_panorama[:origsz[0], :origsz[1]] = back_warp(ims[i], Hs[i], x, y)
            panorama = pyramid_blending(panorama, temp_panorama, mask, levels, 5, 5)

    # slice out the actual image without the extra padding for size
    return panorama[:origsz[0], :origsz[1]].astype(np.float32)


# -------------------------------------------------------------------------------------------------
# ------------------------------------- Bonus Part ------------------------------------------------
# -------------------------------------------------------------------------------------------------

def max_y(im, where):
    """
    returns the index of the first (last) row which is not all 0's.
    :param im: The image to analyze, greyscale
    :param where: 0 to get the first row not all 0's, -1 to get the last
    :return: index of row in the image f the first (last) row which is not all 0's.
    """
    return np.where(np.sum(im, axis=1) != 0.)[0][where]  # rows of not all 0's


def generate_best_mask(warped_corners, curr_pan, added_pan, curr_im_idx, minx, miny, max_cover=False):
    """
    generates the best possible mask (heuristic) for the two given images to stitch together with minimal loss
    :param warped_corners: see get_pan_size_and_borders - return
    :param curr_pan: the current panorama image (n leftmost images), greyscale
    :param added_pan: the new panorama image to merge (the n+1 image), greyscale
    :param curr_im_idx: the index of the current image (of added_pan)
    :param minx, miny: the [x,y] points relative to the center image corresponding to [0,0] location in the panorama
    :param max_cover: if true, forces the mask to not waste any possible covered area (if one of the images cover a
            certain area while the other don't, take this part even the slicing decided otherwise)
    :return: a mask - bool typed array the size of curr_pan, true where curr_pan should be considered,
                    and false otherwise
    """
    # base slicing indices where the overlap between the images occur
    # (hence, where the best slicing path should be found)
    startx = max(int(warped_corners[:, 0, curr_im_idx].min() - minx), 0)
    endx = min(int(warped_corners[:, 0, curr_im_idx-1].max() - minx), curr_pan.shape[1])
    starty = max(int(warped_corners[:, 1, curr_im_idx-1:curr_im_idx+1].min() - miny), 0)
    endy = min(int(warped_corners[:, 1, curr_im_idx-1:curr_im_idx+1].max() - miny), curr_pan.shape[0])

    # find and remove the extra padding rows from bottom and top where both images do not fill (all 0's rows)
    addedim = curr_pan[starty:endy+1, startx:endx+1] + added_pan[starty:endy+1, startx:endx+1]
    first_not_all_throes = max_y(addedim, 0)
    last_not_all_throes = max_y(addedim, -1)
    starty += first_not_all_throes  # can't get out of boundries
    if last_not_all_throes+1 != addedim.shape[0]:  # if there is padding in the bottom
        endy -= (addedim.shape[0]-last_not_all_throes)

    # where there is no overlap, but one the images are present, count as a mistake so it will try not to cut in the
    # middle of such areas
    SINGLE_IMAGE_OVERLAP_MISTAKE_FACTOR = -0.2
    im1, im2 = curr_pan[starty:endy+1, startx:endx+1].copy(), added_pan[starty:endy+1, startx:endx+1].copy()
    im1[np.logical_and(im1 == 0., im2 != 0.)] = SINGLE_IMAGE_OVERLAP_MISTAKE_FACTOR
    im2[np.logical_and(im1 != 0., im2 == 0.)] = SINGLE_IMAGE_OVERLAP_MISTAKE_FACTOR

    # finally, in the processed images, find the best path in which to slice
    # adding 1 to path because everything in the path is brought from the left image
    path = find_best_slice(im1, im2) + 1

    # prepare the final mask
    mask = np.zeros(curr_pan.shape, dtype=np.bool)
    mask[:, :startx+1] = True
    mask[:, endx:] = False
    for i in range(path.size):
        mask[i+starty, :startx + path[i] + 1] = True

    if max_cover:
        mask[np.logical_and(curr_pan == 0., added_pan != 0.)] = False
        mask[np.logical_and(curr_pan != 0., added_pan == 0.)] = True

    return mask


def find_best_slice(im1, im2):
    """
    Use dynamic proggraming in order to find the upward path which slices the two images in such a way that the total
    error is minimal
    :param im1: greyscale image (left image)
    :param im2: greyscale image the size of im1, has overlapping parts with im1
    :return: An array with length equals to the height of im1, each entry marks the rightmost index in the corresponding
            row (row=i -> column=path[i]) which should be taken from im1 (inclusive!)
    """
    height = im1.shape[0]
    E = np.power(im1-im2, 2)

    # dynamically calculate accumulated errors (second row until the end)
    for r in range(1, height):
        E[r, :] += minimum_filter(E[r-1, :], size=1)

    # now backtrace to find ultimate path
    path = np.zeros(height, dtype=np.int)
    path[-1] = np.argmin(E[-1, :])
    for r in range(height-2, -1, -1):
        if path[r+1] == 0:
            # if starting from left edge of the image in next row, only need to account for two possible origin
            # cells above
            s, i = 0, 0  # s is starting pos in current row, i is correction factor to the left
        else:
            s, i = int(path[r+1]-1), 1
        e = int(min(path[r+1]+2, E.shape[1]))  # end is the end pos in current row(can't be larger than width)

        # if upper row left and middle are equal, go straight up
        if E[r, s] == E[r, s+1]:
            path[r] = path[r+1]
        else:
            path[r] = path[r+1] + np.argmin(E[r, s:e]) - i

    return path


def render_panorama_rgb(ims, Hs):
    """
    From a successive list of (rgb) frames, and transformations for the same panorama plane, renders a merged panorama
    image. Uses dynamic programing to find the best slicing path between them (perform on the Y channel in the YIQ
    representation) and use alpha-blending to merge each of the channels before transforming back to RGB
    :param ims: A list of RGB images. (Python list)
    :param Hs: A list of 3x3 homography matrices. Hs[i] is a homography that transforms points from the
                coordinate system of ims [i] to the coordinate system of the panorama. (Python list)
    :return: panorama − A RGB panorama image composed of vertical strips, backwarped using homographies
                    from Hs, one from every image in ims.
    """
    # transform images to YIQ representation
    ims_yiq = [rgb2yiq(im) for im in ims]

    # inits
    alpha_ker_size = 7  # used to blur (gaussian) the final mask to perform alpha-blending
    sz, borders, x, y, warped_corners = get_pan_size_and_borders(ims, Hs)
    panorama = np.zeros((sz[0], sz[1], 3))
    temp_panorama = np.zeros((sz[0], sz[1], 3))

    for i in range(len(ims_yiq)):
        temp_panorama[:] = 0
        if i == 0:
            # backwarp each channel
            for cnl in range(3):
                panorama[:, :, cnl] = back_warp(ims_yiq[i][:,:,cnl], Hs[i], x, y)
        else:
            # backwarp each channel
            for cnl in range(3):
                temp_panorama[:, :, cnl] = back_warp(ims_yiq[i][:, :, cnl], Hs[i], x, y)

            mask = generate_best_mask(warped_corners, panorama[:,:,0], temp_panorama[:,:,0], i, x[0,0], y[0,0],
                                      max_cover=False)
            mask = blur_spatial(mask.astype(np.float32), alpha_ker_size)
            neg_mask = 1 - mask

            # alpha-blend each channel separately
            for cnl in range(3):
                panorama[:,:,cnl] = np.multiply(panorama[:,:,cnl], mask) + np.multiply(temp_panorama[:,:,cnl], neg_mask)

    return clipped_yiq2rgb(panorama)
