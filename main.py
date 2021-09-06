import pydicom
import pydicom as dicom
import PIL # optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut

'''
    dicom_image = dicom.dcmread(image_path)
    im = dicom_image.pixel_array.astype(float)

    rescaled_image = (np.maximum(im,0)/im.max())*255 # float pixel
    final_image = np.uint8(rescaled_image)
    final_image = Image.fromarray(final_image)
    print(final_image)
    final_image.show()
'''
def make_png():
    path = 'G:/Contrast-enhanced Problem/Nect CT/001950685/001950685 axi ce/'

    image_path = path + '001950685 axi ce0001.dcm'

    window_center = -600
    window_width = 1600

    slice = pydicom.read_file(image_path)
    # print(slice.shape)
    s = int(slice.RescaleSlope)
    b = int(slice.RescaleIntercept)
    image = s * slice.pixel_array + b

    # apply_voi_lut( )
    slice.WindowCenter = window_center
    slice.WindowWidth = window_width
    print(slice.WindowWidth, slice.WindowHeight)
    image2 = apply_voi_lut(image, slice)
    print(image2)
    print(image2.shape)
    plt.axis('off')
    plt.imshow(image2, cmap='gray')
    plt.savefig('C:/Users/dongyoung/Desktop/A.png')

def show_png():
    path = 'C:/Users/dongyoung/Desktop/A.png'
    im = Image.open(path)
    im = im.resize((600, 600))
    print(im.size)
    im.show()


if __name__ == '__main__':
    show_png()
