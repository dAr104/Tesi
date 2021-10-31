import numpy as np
from scipy.ndimage import sobel
from scipy.stats import moment
from PIL import ImageOps, ImageFilter


def image_array_in_greyscale(img):
    """Function that take a image's path and convert that image in a greyscale array"""

    img_grey = ImageOps.grayscale(img)
    data = np.asarray(img_grey) / 255
    return data


def feature_directionality(img, n=12):

    array_greyscale = image_array_in_greyscale(img)
    gx = sobel(array_greyscale, 1)  # horizontal gradient
    gy = sobel(array_greyscale, 0)  # vertical gradient
    theta = np.arctan2(gy, gx)
    discrete_thetas = ((theta + np.pi) / (2 * np.pi)) * n
    x = discrete_thetas.astype(int) % n
    p_theta = np.bincount(x.flatten(), minlength=12) / x.size

    return ((n / (n - 1)) * ((p_theta - 1/n) ** 2).sum()) ** (1/2)


def _cooccurrence_matrix_dir(values, bins, di, dj):
    """Helper for the computation of the co-occurrence matrix.

    (di, dj) is the spatial displacement of neighbor pixels.  The
    elements of values must be integers in the range [0, 1, ..., bins)

    """
    m, n = values.shape
    if dj > 0:
        codes = values[:m + di, :n - dj] + bins * values[-di:, dj:] # di è <= 0 per costruzione, dj > 0
    else:
        codes = values[:m + di, :n + dj] + bins * values[-di:, - dj:]
    entries = np.bincount(codes.ravel(), minlength=bins ** 2)
    return entries.reshape(bins, bins)


def feature_line_likeliness(img, n=12, d=4):

    array_greyscale = image_array_in_greyscale(img)
    d_vector = [[0, d], [-d, d], [-d, 0], [-d, -d]]
    gx = sobel(array_greyscale, 1)  # horizontal gradient
    gy = sobel(array_greyscale, 0)  # vertical gradient
    theta = np.arctan2(gy, gx)
    discrete_thetas = ((theta + np.pi) / (2 * np.pi)) * n
    x = discrete_thetas.astype(int) % n

    ij = np.arange(n)
    m = abs(np.cos((ij[None, :] - ij[:, None]) * ((2 * np.pi) / n)))

    f_lin_k = 0
    for k in range(d):
        cm_d = _cooccurrence_matrix_dir(x, n, d_vector[k][0], d_vector[k][1])
        f_lin_k += (m * cm_d).sum() / cm_d.sum()

    return f_lin_k * 0.25


def addAtPos(mat1, mat2, xypos):
    """
    Add two matrices of different sizes in place, offset by xy coordinates
    Usage:
      - mat1: base matrix
      - mat2: add this matrix to mat1
      - xypos: tuple (x,y) containing coordinates
    """
    y, x = xypos
    ysize, xsize = mat2.shape
    xmax, ymax = (x + xsize), (y + ysize)
    mat1[y:ymax, x:xmax] = mat2
    return mat1


def feature_coarseness(img, K=4):
    img_grey = ImageOps.grayscale(img)

    H = np.asarray(img_grey).shape[0]
    W = np.asarray(img_grey).shape[1]

    ak = []
    for k in range(1, K + 1):
        radius = ((2 ** k) - 1) / 2
        tmp_img = img_grey.filter(ImageFilter.BoxBlur(radius))
        data = np.asarray(tmp_img) / 255

        #H/V difference

        i = 2 ** (k-1)
        vdiff = np.abs(data[:-(2 * i), :] - data[2 * i:, :])
        hdiff = np.abs(data[:, :-(2 * i)] - data[:, 2 * i:])

        #padding

        vdiff_reshape = addAtPos(np.full((H, W), -np.inf), vdiff, (2**(k-1), 0))
        hdiff_reshape = addAtPos(np.full((H, W), -np.inf), hdiff, (0, 2**(k-1)))

        ak.append(np.maximum(hdiff_reshape, vdiff_reshape)) # prendo i valori massimi tra le due "maschere"

    # K_ è l'array con i valori massimi delle K=4 iterazioni
    allK = np.stack(ak, -1)
    K_ = 1 + np.argmax(allK, -1)  # + 1 since np indexes start from 0
    f_crs = (2 ** K_).mean() / (2 ** K)
    return f_crs


def feature_roughness(img):
    array_greyscale = image_array_in_greyscale(img)
    height = array_greyscale.shape[0]
    widgt = array_greyscale.shape[1]
    r = np.average(array_greyscale)
    result = ((array_greyscale - r) ** 2).sum()
    f_roughness = 2 * ((1 / (height * widgt)) * result) ** (1/2)
    return f_roughness


def feature_contrast(img, n=1 / 4):
    array_greyscale = image_array_in_greyscale(img)
    mu4 = moment(array_greyscale, moment=4, axis=None)  # momento centrale del 4° ordine
    std = np.std(array_greyscale)
    if std == 0:
        return 0.
    kurtosis = mu4 / (std ** 4)
    return 2 * (std / (kurtosis ** n))