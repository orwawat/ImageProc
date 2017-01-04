import sol4
from sol4_utils import read_image
import matplotlib.pyplot as plt

def test_harris_detector():
    im = read_image('external/backyard1.jpg', 1)
    res = sol4.harris_corner_detector(im)
    plt.imshow(im, cmap=plt.cm.gray)
    plt.scatter(res[1], res[0])
    plt.show(block=True)

def main():
    print("Testing sol4. starting")
    try:
        for test in [test_harris_detector]:
            test()
    except Exception as e:
        print("Tests failed. error: {0}".format(e))
        exit(-1)
    print("All tests passed!")

if __name__ == '__main__':
    main()