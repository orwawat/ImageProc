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
    from sol4 import sample_descriptor, map_coord_2_level
    m = np.arange(100**2).reshape((100,-1))
    pyr, f = build_gaussian_pyramid(m, 3, 3)
    # pos = np.array([[9,9],[50,70]])
    pos = map_coord_2_level(np.array([[15,15],[50,70]]))
    desc = sample_descriptor(pyr[2], pos, 3)
    if np.any(desc[:,:,0] !=desc[:,:,1]): raise Exception('Bad desc')
    print('OK!!!!!!')

def test_desc():
    from sol4 import map_coord_2_level, sample_descriptor, get_windows_coords
    from scipy.ndimage import map_coordinates
    im1 = read_image('external/oxford1.jpg', 1)
    pyr1, f = build_gaussian_pyramid(im1, 3, 3)
    pos1, desc1 = sol4.find_features(pyr1)
    ind = np.random.choice(np.arange(pos1.shape[0]), 15)
    pos2 = map_coord_2_level(pos1[ind])
    desc2 = sample_descriptor(pyr1[2], pos2, 3)
    if np.any(desc1[:,:,ind] != desc2): raise Exception("Wrong desc")

    coord_window = get_windows_coords(pos2[0][np.newaxis,:], 3)
    desc = map_coordinates(pyr1[2], coord_window)
    desc -= np.mean(desc)
    desc /= np.linalg.norm(desc)
    if not np.isclose(1., np.dot(desc, desc1[:,:,ind[0]].flatten())): raise Exception("Bad dot: {0}".format( np.dot(desc, desc1[:,:,ind[0]].flatten())))

    pos2 = map_coord_2_level(pos1)
    desc2 = sample_descriptor(pyr1[2], pos2, 3)
    if np.any(desc1[:, :, :] != desc2[:,:,:]): raise Exception("Wrong desc 2")
    print('ok')

def test_apply_hom():
    from sol4 import apply_homography
    print('Test hom')
    im1 = read_image('external/oxford1.jpg', 1)
    pyr1, f = build_gaussian_pyramid(im1, 3, 3)
    pos1, desc1 = sol4.find_features(pyr1)
    H = np.eye(3)
    pos2 = apply_homography(pos1, H)
    if np.any(pos1!=pos2): raise Exception("bad hommie")
    print('Hom ok')

def test_disp_points():
    from sol4 import map_coord_2_level, sample_descriptor, get_windows_coords
    from scipy.ndimage import map_coordinates
    im1 = read_image('external/oxford1.jpg', 1)
    pyr1, f = build_gaussian_pyramid(im1, 3, 3)
    pos1, desc1 = sol4.find_features(pyr1)
    ind = np.random.choice(np.arange(pos1.shape[0]), 15)
    pos2 = map_coord_2_level(pos1[ind])
    plt.figure()
    plt.subplot(1,2,1); plt.imshow(pyr1[0],cmap=plt.cm.gray); plt.scatter(pos1[ind][:,1], pos1[ind][:,0])
    plt.subplot(1,2,2); plt.imshow(pyr1[2],cmap=plt.cm.gray); plt.scatter(pos2[:,1], pos2[:,0])

    plt.figure()
    for i in range(1,16):
        plt.subplot(3,5,i); plt.imshow(desc1[:,:,ind[i-1]],cmap=plt.cm.gray); plt.title('X:{0}, Y:{1}'.format(pos2[i-1,1], pos2[i-1,0]));
    plt.show(block=True)

def test_matches():
    im1 = read_image('external/oxford1.jpg', 1)
    # im1 = read_image('external/oxford1.jpg', 1)[140:300,420:600]
    im2 = read_image('external/oxford2.jpg', 1)
    # im2 = read_image('external/oxford2.jpg', 1)[180:320,140:300]
    pyr1, f = build_gaussian_pyramid(im1,3,3)
    pyr2, f = build_gaussian_pyramid(im2,3,3)
    pos1, desc1 = sol4.find_features(pyr1)
    pos2, desc2 = sol4.find_features(pyr2)

    # pos1=np.fliplr(pos1)
    # pos2=np.fliplr(pos2)

    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(im1, cmap=plt.cm.gray)
    plt.scatter(pos1[:,0], pos1[:,1])
    plt.subplot(2, 2, 2)
    plt.imshow(im2, cmap=plt.cm.gray)
    plt.scatter(pos2[:, 0], pos2[:, 1])

    ind1, ind2 = sol4.match_features(desc1, desc2, 0.7)
    mpos1, mpos2 = pos1[ind1], pos2[ind2]

    plt.subplot(2, 2, 3)
    plt.imshow(im1, cmap=plt.cm.gray)
    plt.scatter(mpos1[:, 0], mpos1[:, 1])
    plt.subplot(2, 2, 4)
    plt.imshow(im2, cmap=plt.cm.gray)
    plt.scatter(mpos2[:, 0], mpos2[:, 1])

    # for i1, i2 in zip(ind1, ind2):
    #     print('Ind1: {0}, Ind2: {1}, DescSim: {2}'.format(i1, i2, np.dot(desc1[:,:,i1].flatten(),
    #                                                                          desc2[:,:,i2].flatten())))

    H, inliers = sol4.ransac_homography(mpos1, mpos2, 500, 7)
    sol4.display_matches(im1, im2, mpos1, mpos2, inliers)

def main():
    # test_disp_points()
    # test_desc()
    # test_sample_desc()
    # test_harris_detector()
    # d = np.arange(100*2).reshape((2,10,10)).transpose((1,2,0))
    # dt = d.transpose((2, 0, 1)).reshape((d.shape[2], -1))
    # plt.figure()
    # plt.subplot(2,2,1)
    # plt.imshow(d[:,:,0])
    # plt.subplot(2, 2, 2)
    # plt.imshow(d[:, :, 1])
    # plt.subplot(2, 1, 2)
    # plt.imshow(dt)
    # plt.show(block=True)
    print("Testing sol4. starting")
    try:
        for test in [test_desc,test_apply_hom, test_matches]:
            test()
    except Exception as e:
        print("Tests failed. error: {0}".format(e))
        exit(-1)
    print("All tests passed!")

if __name__ == '__main__':
    main()