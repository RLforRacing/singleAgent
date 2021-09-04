import numpy as np
import scipy.ndimage as snd

def rgb_to_gray(img):
    return img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114

def detect_edge(img, threshold=10.0):
    """
    Returns black and white image with edges.
    Uses Sobel-filter in x-y directions + thresholding.
    """
    img = rgb_to_gray(img)
    img_sobel_x = snd.filters.sobel(img, axis=0)
    img_sobel_y = snd.filters.sobel(img, axis=1)
    grad_img = np.sqrt(np.square(img_sobel_x) + np.square(img_sobel_y))
    sbl_max = np.amax(abs(grad_img))
    bn_img = np.abs(grad_img) >= (sbl_max / threshold)
    return bn_img.astype(int) * 255

def locate_car(img):
    """
    Locates the center-of-mass of the car based on its known RGB value.
    There are some curbs with red in it which can be an issue.
    TODO: handle red color in curbs
    """
    red = (img[:, :, 0] == 204).astype(int) * 255
    if np.sum(red) == 0:
        return -1, -1
    else:
        return snd.center_of_mass(red)

def dist_from_edge_ahead(img):
    """
    Returns the distance from the edge of the road ahead of the car, in pixels.
    If no edge is detected ahead or car cannot be located returns -1.
    TODO: find car length exactly
    """
    edges = detect_edge(img)
    car_x, car_y = locate_car(img)
    if car_x == -1 or car_y == -1:
        return -1
    col_ahead = edges[:, int(car_y)]
    im_height = col_ahead.shape[0]
    col_ahead = np.flip(col_ahead)
    flipped_car_x = int(im_height - car_x)
    col_ahead[: flipped_car_x + 6] = 0  # 6 is a guess for the car length
    print(col_ahead)
    indx_road = np.argmax(col_ahead)
    return indx_road - flipped_car_x if indx_road > 0 else -1

def strip_indicators(img):
    """
    Removes the indicator bar from the observations to clear up the image.
    """
    return img[:84, :, :]