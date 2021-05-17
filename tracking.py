import setup_path
import cv2 as cv
import numpy as np


def tracking(mapimg, x_val, y_val):
    img_y, img_x = np.shape(mapimg)

    x_len = 22
    y_len = 45

    x_block = round(np.shape(mapimg)[1]/x_len, 1)
    y_block = round(np.shape(mapimg)[0]/y_len, 1)

    x = round(x_val, 1) + 11
    y = round(y_val, 1) + 42

    x = int(x * x_block)
    y = int(y * y_block)

    x = 2 if x-2 < 0 else x
    x = img_x-3 if x+2 > img_x-3 else x
    y = 2 if y-2 < 0 else y
    y = img_y-3 if y+2 > img_y-3 else y

    mapimg[y-2:y+2, x-2:x+2] = 255

    return mapimg
