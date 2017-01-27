from scipy import ndimage


def blur(img, sigma=10):
    return ndimage.filters.gaussian_filter(img, sigma)