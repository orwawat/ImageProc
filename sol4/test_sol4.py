import sol4
from sol4_utils import read_image, build_gaussian_pyramid
import matplotlib.pyplot as plt
from sol4_add import spread_out_corners

def test_harris_detector():
    im = read_image('external/backyard1.jpg', 1)
    plt.figure()
    res = spread_out_corners(im,7,7,15)
    plt.imshow(im, cmap=plt.cm.gray)
    plt.scatter(res[:, 0], res[:, 1])
    plt.show(block=True)
    for pth in ['external/office1.jpg', 'external/office2.jpg']:
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

def test_matches():
    im1 = read_image('external/office1.jpg', 1)
    im2 = read_image('external/office2.jpg', 1)
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

    ind1, ind2 = sol4.match_features(desc1, desc2, 0.1)
    mpos1, mpos2 = pos1[ind1], pos2[ind2]

    plt.subplot(2, 2, 3)
    plt.imshow(im1, cmap=plt.cm.gray)
    plt.scatter(mpos1[:, 0], mpos1[:, 1])
    plt.subplot(2, 2, 4)
    plt.imshow(im2, cmap=plt.cm.gray)
    plt.scatter(mpos2[:, 0], mpos2[:, 1])

    plt.show()
    H, inliers = sol4.ransac_homography(mpos1, mpos2, 15, 5)
    sol4.display_matches(im1, im2, mpos1, mpos2, inliers)

def main():
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