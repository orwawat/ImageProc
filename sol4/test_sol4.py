import sol4
from sol4_utils import read_image, build_gaussian_pyramid
import matplotlib.pyplot as plt
from sol4_add import spread_out_corners
import numpy as np

def test_harris_detector():
    im = read_image('external/oxford2.jpg', 1)
    plt.figure()
    res = spread_out_corners(im,7,7,15)
    plt.imshow(im, cmap=plt.cm.gray)
    plt.scatter(res[:, 0], res[:, 1])
    # plt.show(block=True)
    for pth in ['external/oxford2.jpg','external/office1.jpg', 'external/office2.jpg']:
        plt.figure()
        im = read_image(pth, 1)
        res = sol4.harris_corner_detector(im)
        plt.imshow(im, cmap=plt.cm.gray)
        plt.scatter(res[:,0], res[:,1])

        pyr, f = build_gaussian_pyramid(im, 3, 3)
        plt.figure()
        im=pyr[2]
        res = sol4.harris_corner_detector(im)
        plt.imshow(im, cmap=plt.cm.gray)
        plt.scatter(res[:, 0], res[:, 1])
    plt.show(block=True)

def test_sample_desc():
    from sol4 import sample_descriptor
    m = np.arange(100**2).reshape((100,-1))
    pos = np.array([[9,9],[50,70]])
    desc, pos2 = sample_descriptor(m, pos, 3)
    if np.any(pos !=pos2): raise Exception('Bad pos')
    if np.any(desc[:,:,0] !=m[6:13,6:13]): raise Exception('Bad desc1')
    if np.any(desc[:,:,1] !=m[47:54,67:74]): raise Exception('Bad desc2')


def test_matches():
    im1 = read_image('external/oxford1.jpg', 1)
    # im1 = read_image('external/oxford1.jpg', 1)[140:300,420:600]
    im2 = read_image('external/oxford2.jpg', 1)
    # im2 = read_image('external/oxford2.jpg', 1)[180:320,140:300]
    pyr1, f = build_gaussian_pyramid(im1,3,3)
    pyr2, f = build_gaussian_pyramid(im2,3,3)
    pos1, desc1 = sol4.find_features(pyr1)
    pos2, desc2 = sol4.find_features(pyr2)
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(im1, cmap=plt.cm.gray)
    plt.scatter(pos1[:,0], pos1[:,1])
    plt.subplot(2, 2, 2)
    plt.imshow(im2, cmap=plt.cm.gray)
    plt.scatter(pos2[:, 0], pos2[:, 1])

    ind1, ind2 = sol4.match_features(desc1, desc2, 0.5)
    mpos1, mpos2 = pos1[ind1], pos2[ind2]

    plt.subplot(2, 2, 3)
    plt.imshow(im1, cmap=plt.cm.gray)
    plt.scatter(mpos1[:, 0], mpos1[:, 1])
    plt.subplot(2, 2, 4)
    plt.imshow(im2, cmap=plt.cm.gray)
    plt.scatter(mpos2[:, 0], mpos2[:, 1])

    H, inliers = sol4.ransac_homography(mpos1, mpos2, 150, 3)
    sol4.display_matches(im1, im2, mpos1, mpos2, inliers)

def main():
    # test_sample_desc()
    # test_harris_detector()
    print("Testing sol4. starting")
    try:
        for test in [test_matches]:
            test()
    except Exception as e:
        print("Tests failed. error: {0}".format(e))
        exit(-1)
    print("All tests passed!")

if __name__ == '__main__':
    main()