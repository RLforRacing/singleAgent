import numpy as np
import scipy.ndimage as snd
from PIL import Image

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


def locate_off_road(img):
    """
    Locates the off road areas (ie, the green portions) of the image based on RGB value
    RGBna: image where green pixels are turned black
    green: locations of green pixels
    """
    RGBim = Image.fromarray(img)
    HSVim = RGBim.convert('HSV')

    # Make numpy versions
    RGBna = np.array(img)
    HSVna = np.array(HSVim)

    # Extract Hue
    H = HSVna[:,:,0]

    # Find all green pixels, i.e. where 100 < Hue < 140
    lo,hi = 100,140
    # Rescale to 0-255, rather than 0-360 because we are using uint8
    lo = int((lo * 255) / 360)
    hi = int((hi * 255) / 360)
    green = np.where((H>lo) & (H<hi))
    #not_green = np.where(H<lo)

    # Make all green pixels black in original image
    RGBna[green] = [0,0,0]


    return RGBna, green



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
    # print(col_ahead)
    indx_road = np.argmax(col_ahead)
    return indx_road - flipped_car_x if indx_road > 0 else -1

def dist_from_right_and_left(img):
    """
    Returns the distance from the right and left edge of the road, in pixels.
    If no edge is detected ahead or car cannot be located returns -1.
    """
    edges = detect_edge(img)
    car_x, car_y = locate_car(img)
    if car_x == -1 or car_y == -1:
        return -1, -1
    row_cur = edges[int(car_x), :]
    left_side = np.flip(row_cur[:int(np.floor(car_y))-3])
    right_side = row_cur[int(np.ceil(car_y))+3:]
    # print(left_side)
    # print(right_side)
    return np.argmax(left_side), np.argmax(right_side)

def strip_indicators(img):
    """
    Removes the indicator bar from the observations to clear up the image.
    """
    return img[:84, :, :]