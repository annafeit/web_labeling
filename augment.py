from scipy import ndimage
import numpy as np

def blur(image, sigma=1):
    """
    Gaussian filter to blur the image
    """
    img = np.copy(image)
    img[0,:,:] = ndimage.filters.gaussian_filter(img[0,:,:], sigma)
    img[1,:,:] = ndimage.filters.gaussian_filter(img[1,:,:], sigma)
    img[2,:,:] = ndimage.filters.gaussian_filter(img[2,:,:], sigma)
    return img 

def mirror_horizontal(img):
    """
    Mirror given image horizontally
    """
    return np.fliplr(img.transpose(1,2,0)).transpose(2,0,1)

def weightedAverage(pixel):
    """
    computes the weighted average of a given RGB pixel. https://en.wikipedia.org/wiki/Grayscale
    """
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

def greyscale(img):
    """
    Computes the corresponding greyscale from the given image. 
    Note: The returned image has of course only 1 channel so the format of the image changes!
    """
    image = img.transpose(1,2,0)
    grey = np.zeros((image.shape[0], image.shape[1])) # init 2D numpy array
    # get row number
    for rownum in range(image.shape[0]):
        for colnum in range(image.shape[1]):
            grey[rownum][colnum] = weightedAverage(image[rownum,colnum])
    return grey