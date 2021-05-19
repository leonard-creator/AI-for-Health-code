'''
Pre Processing retinal Fundus Images
Detecting and masking the retinal Fundus Disk
by Leonard Tiling
'''

import os
import pandas as pd
import numpy as np
from PIL import Image
from skimage import measure, io
from skimage.util import img_as_ubyte
from skimage.feature import canny


import scipy.ndimage as ndimage
import scipy.ndimage as ndimage


#Works on: Messidor/all_cropped_ , palm/all_cropped , RIADD/all_cropped


def mask(filename, outdir=None):
    dir = '/data/project/retina/Messidor/all_cropped_margins' + '/' + filename

    # load and convert image to uint8 and 2D Grayscale
    img = Image.open(dir)
    np_img = np.asarray(img)
    ret = img_as_ubyte(np.copy(np_img))  # bc np_img is read only
    color_retina = ret
    retina = ret[:, :, 0]  # gray

    # creating edges from retina
    edges = canny(retina, sigma=2, low_threshold=10, high_threshold=50)

    # find contour in edges
    c = measure.find_contours(edges, 0, 'high')
    contour = sorted(c, key=lambda x: len(x))[-1]  # convert tuple to integer vor mask[...] operation

    mask = np.zeros_like(retina, dtype='bool')
    # Create a contour image by using the contour coordinates rounded to their nearest integer value
    mask[(contour[:, 0]).astype('uint'), (contour[:, 1]).astype('uint')] = 1
    # Fill in the hole created by the contour boundary
    mask = ndimage.binary_fill_holes(mask)
    # Invert the mask since you want pixels outside of the region
    mask = ~mask
    # masking image

    color_retina[mask] = 0 # black

    io.imsave(outdir + filename, color_retina)

def cheapMask(filename, outdir, dict):

    # load and convert for canny
    dir = '/data/project/uk_bb/cvd/data/ukb_downloads/updated_showcase_43098/ukb_bulk/all_retina_bulk_cropped_margins' + '/' + filename
    img = Image.open(dir)
    np_img = np.asarray(img)
    ret = img_as_ubyte(np.copy(np_img))  # bc np_img is read only
    retina = ret[:, :, 0]  # gray
    # retina = color.rgb2gray(ret)

    mask = retina < 20  # detect all nearly black arreas

    ret[mask] = 0  # choose value vor masked points
    if mask.sum() <= 420000:
        io.imsave(outdir + filename, ret)
        dict[filename] = [mask.sum(),1]
    else:
        dict[filename] = [mask.sum(),0]



if __name__ == '__main__':

    directory = '/data/project/uk_bb/cvd/data/ukb_downloads/updated_showcase_43098/ukb_bulk/all_retina_bulk_cropped_margins'
    outdirectory =  '/data/analysis/ag-reils/ag-reils-shared-students/tilingl/code/RetinalFundusSSL/PreCropping/b_img/'
    os.makedirs(outdirectory, exist_ok=True)

    c =0
    # dic to loop values in, transform to pandas.dataframe at the end
    data_values = {}
    for filename in os.listdir(directory):
        c += 1
        if(c == 10):
           break
        cheapMask(filename,outdirectory,data_values)

    # transforming list into dataframe and csv
    df = pd.DataFrame.from_dict(data_values, orient='index', columns=['mask.sum_value', 'saved?'])
    df.to_csv('/data/analysis/ag-reils/ag-reils-shared-students/tilingl/code/RetinalFundusSSL/PreCropping/blackV.csv', index=True)
