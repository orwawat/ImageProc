import matplotlib.pyplot as plt
import numpy as np
import os
import sol4
import sol4_utils


def generate_panorama(data_dir, file_prefix, num_images, pan_gen, figsize=(20, 20)):
    """
    Geneare panorama out of the files in the given dir with the given prefix.
    displays the results
    :param data_dir: The directory where the image are
    :param file_prefix: The prefix for each image (convention is nameN)
    :param num_images: how many images to render from the series
    :param pan_gen: A function which recieves ims_rgb and Htot (Accumulkated homographies) and use them to generate and
                  return a rgb panorama image
    :param figsize: The figure size of the final panorama
    """
    # The naming convention for a sequence of images is nameN.jpg, where N is a running number 1,2,..
    files = [os.path.join(data_dir, '%s%d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]

    # Read images.
    ims = [sol4_utils.read_image(f, 1) for f in files]

    # Extract feature point locations and descriptors.
    def im_to_points(im):
        pyr, _ = sol4_utils.build_gaussian_pyramid(im, 3, 7)
        return sol4.find_features(pyr)

    p_d = [im_to_points(im) for im in ims]

    # Compute homographies between successive pairs of images.
    Hs = []
    for i in range(num_images - 1):
        points1, points2 = p_d[i][0], p_d[i + 1][0]
        desc1, desc2 = p_d[i][1], p_d[i + 1][1]

        # Find matching feature points.
        ind1, ind2 = sol4.match_features(desc1, desc2, .8)
        points1, points2 = points1[ind1, :], points2[ind2, :]

        # Compute homography using RANSAC.
        H12, inliers = sol4.ransac_homography(points1, points2, 10000, 8)

        # Display inlier and outlier matches.
        sol4.display_matches(ims[i], ims[i + 1], points1, points2, inliers=inliers)
        Hs.append(H12)

    # Compute composite homographies from the panorama coordinate system.
    Htot = sol4.accumulate_homographies(Hs, (num_images - 1) // 2)

    # Final panorama is generated using 3 channels of the RGB images
    ims_rgb = [sol4_utils.read_image(f, 2) for f in files]

    # generate the panorama
    panorama = pan_gen(ims_rgb, Htot)

    # plot the panorama
    plt.figure(figsize=figsize)
    plt.imshow(panorama.clip(0, 1))
    plt.show()


def gen_pan_with_pyr_blend(ims_rgb, Htot):
    """
    Renders rgb panorama with pyramid blending, for each channel separately
    :param ims_rgb: a list of consecutive rgb images
    :param Htot: a list of corresponding homographies from image i to the center image
    :return: A rgb image with the rendered panorama
    """
    # Render panorama for each color channel and combine them.
    panorama = [sol4.render_panorama([im[..., i] for im in ims_rgb], Htot) for i in range(3)]
    panorama = np.dstack(panorama)
    return panorama


def gen_pan_with_dynamic_stiching(ims_rgb, Htot):
    """
    Renders rgb panorama with dynamic programming stiching. Find stich in YIQ and sue it for all channels.
    :param ims_rgb: a list of consecutive rgb images
    :param Htot: a list of corresponding homographies from image i to the center image
    :return: A rgb image with the rendered panorama
    """
    # generate panorama using dynamic proggramming stiching
    panorama = sol4.render_panorama_rgb(ims_rgb, Htot)
    return panorama


def main():
    generate_panorama('external/', 'shuk', 4, gen_pan_with_pyr_blend)
    generate_panorama('external/', 'morning_shuk', 4, gen_pan_with_dynamic_stiching)

    # TODO - add parallax images

    # TODO - delete
    generate_panorama('external/', 'oxford', 2, gen_pan_with_pyr_blend)
    generate_panorama('external/', 'backyard', 3, (20, 10), gen_pan_with_pyr_blend)
    generate_panorama('external/', 'office', 4, gen_pan_with_pyr_blend)


if __name__ == '__main__':
    main()
