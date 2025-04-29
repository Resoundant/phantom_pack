import cv2
import numpy as np
import matplotlib.pyplot as plt
from phantom_pack import find_circles

def create_test_image() -> np.ndarray:
    height = 256
    width = 256
    channels = 1
    radius = 9
    spacing = 4*radius
    x_offset = radius*2
    y_offset = radius*2
    img = np.zeros((height, width , channels), dtype=np.uint8)
    for i in range(7):
        for j in range(7):
            x = int(x_offset + j*spacing)
            y = int(y_offset + i*spacing)
            intensity = int(255/49 * (i*7 + j))
            cv2.circle(img, (x,y), radius, (intensity), -1) # solid circle (thickness = -1) filled with  1
    return img

if __name__ == "__main__":
    test_image = create_test_image()
    # cv2.imshow("Test Image", test_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    circles = find_circles(test_image, minDist=1, minRadius=1, maxRadius=20)
    x=1