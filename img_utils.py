import cv2
import os
import numpy as np
import warnings
from PIL import ImageEnhance, Image
from imutils import rotate, translate
import sys; sys.path.append("../")
from src.nms import nms
from glob import glob
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage import transform as tf


def read(img_dir, rgb=True):
    if not os.path.exists(img_dir):
        raise ValueError("Given directory does not exist! [" + img_dir + "]")

    img = cv2.imread(img_dir)

    if img is None:
        raise ValueError("Cannot read the image in the given directory!")

    if rgb:
        img = bgr2rgb(img)

    return img


def bgr2rgb(img):
    return img[:, :, (2, 1, 0)]


def augment(img, label, attention):

    ## FOR RMIT DATASET PLEASE UNCOMMENT THE FOLLOWING PART!
    """
    # geometric
    stage1 = [no_augmentation, horizontal_flip, vertical_flip]
    img, label, attention = stage1[np.random.randint(0, high=len(stage1))](img, label, attention)

    stage2 = [random_translation, random_rotation, no_augmentation, swap_sides, random_swap]
    img, label, attention = stage2[np.random.randint(0, high=len(stage2))](img, label, attention)

    # perceptual
    stage3 = [random_brightness, random_contrast, random_saturation, histogram_equalization,
              blur, no_augmentation]

    img, label, attention = stage3[np.random.randint(0, high=len(stage3))](img, label, attention)

    """
    ## FOR ENDOVIS DATASET PLEASE UNCOMMENT THE FOLLOWING PART!

    """
    # geometric
    stage1 = [no_augmentation, horizontal_flip, vertical_flip]
    img, label, attention = stage1[np.random.randint(0, high=len(stage1))](img, label, attention)

    stage2 = [random_translation, random_rotation, no_augmentation, swap_sides, random_swap]
    img, label, attention = stage2[np.random.randint(0, high=len(stage2))](img, label, attention)
    """

    raise NotImplementedError("Please choose data augmentation strategy!")

    return img, label, attention


def no_augmentation(img, label, attention):
    return img, label, attention


def fancy_pca(img, label, attention):

    img = img.astype(np.float32)
    normalize = False
    if img.max() > 1:
        img /= 255.
        normalize = True

    org_shape = img.shape
    img.shape = (-1, 3)
    mu = img.mean(axis=0)
    img_c = img - mu

    cov = np.cov(img_c, rowvar=False)
    lambdas, p = np.linalg.eig(cov)
    alphas = np.random.normal(0, .1, 3)

    noise = np.dot(p, alphas*lambdas)

    img += noise
    img.shape = org_shape
    if normalize:
        img *= 255.
        img = np.clip(img, 0, 255)
    else:
        img = np.clip(img, 0, 1)

    return img, label, attention


def tps(img, label, attention):
    if np.sum(label) == 0:
        return img, label, attention

    tps = cv2.createThinPlateSplineShapeTransformer()
    multi = False
    points, _ = nms(label[:, :, :5])
    if len(points[0]) > 1:
        multi = True

    p_ = []
    for p in points:
        for e in list(map(list, p)):
            p_.append(e)
    points = np.array(p_).reshape(1, -1, 2)

    t_points = points.copy()
    t_points[:, 0, 0] -= np.random.randint(-2, 2, size=1)
    t_points[:, 0, 1] -= np.random.randint(-2, 2, size=1)
    if multi:
        t_points[:, 1, 0] -= np.random.randint(-2, 2, size=1)
        t_points[:, 1, 1] -= np.random.randint(-2, 2, size=1)
    matches = [cv2.DMatch(i, i, 0) for i in range(5)]

    tps.estimateTransformation(t_points, points, matches)

    return tps.warpImage(img), tps.warpImage(label), attention


def elastic_deform(img, label, attention):
    random_state = np.random.RandomState(None)
    shape = img.shape[:2]
    alpha = 34
    sigma = 4

    for ch in range(img.shape[-1]):
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma=sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma=sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        img[:, :, ch] = map_coordinates(img[:, :, ch], indices, order=1).reshape(shape)

    return img, label, attention


def random_swap(img, label, attention):

    if img.shape[-1] > 3:
        return img, label, attention

    if np.sum(label) == 0:
        return img, label, attention

    root = "/media/digits2/9baf4857-aa15-462d-b5db-022aa5cc3dc5/MicrosurgicalTooltracking/data/EndoVis/"
    imgs = glob(root + "labelled_train/*.jpg")
    idx = np.random.randint(0, len(imgs))
    random_img = read(imgs[idx])
    random_img = cv2.resize(random_img, (img.shape[1], img.shape[0]))
    fname = imgs[idx].split("/")[-1]
    fname = fname.split(".")[0]
    random_label = np.load(root + "training_labels/" + fname + ".npy")
    random_label = cv2.resize(random_label, (img.shape[1], img.shape[0]))

    peaks_random, _ = nms(random_label[:, :, :5])
    peaks, _ = nms(label[:, :, :5])

    if len(peaks[0]) > 1:
        x1, _ = peaks[0][0]
        x2, _ = peaks[0][1]
        x = (x1 + x2) // 2
    else:
        x1, _ = peaks[0][0]
        try:
            if peaks[3][0][0] > img.shape[1] // 2:
                x = x1 - 20
            else:
                x = x1 + 20
        except:
            x = x1

    if len(peaks_random[0]) > 1:
        x1_r, _ = peaks_random[0][0]
        x2_r, _ = peaks_random[0][1]
        x_r = (x1_r + x2_r) // 2
    else:
        x1_r, _ = peaks[0][0]
        try:
            if peaks_random[3][0][0] > img.shape[1] // 2:
                x_r = x1_r - 20
            else:
                x_r = x1_r + 20
        except:
            x_r = x1_r

    w = img.shape[1]
    random_patch = random_img[:, :w-x_r, :].astype(np.float32)
    random_patch_label = random_label[:, :w-x_r, :]

    img[:, :w - x_r, :] = random_patch
    if x_r > x:
        img[:, w-x:w-x_r, :] = 0
    elif x_r < x:
        img[:, w-x_r:w:x, :] = 0
    label[:, :w - x_r, :] = random_patch_label
    if x_r > x:
        label[:, w-x:w-x_r, :] = 0
    elif x_r < x:
        label[:, w-x_r:w:x, :] = 0

    return img, label, attention


def swap_sides(img, label, attention):

    if img.shape[-1] > 3:
        return img, label, attention

    if np.sum(label) == 0:
        return img, label, attention

    peaks, _ = nms(label[:, :, :5])
    w = img.shape[1]

    if len(peaks[0]) > 1:
        x1, _ = peaks[0][0]
        x2, _ = peaks[0][1]
        x = (x1 + x2) // 2
    else:
        x1, _ = peaks[0][0]
        try:
            if peaks[3][0][0] > img.shape[1] // 2:
                x = x1 - 20
            else:
                x = x1 + 20
        except:
            x = x1

    tmp = img.copy()
    img[:, :w-x, :] = tmp[:, x:, :]
    img[:, w-x:, :] = tmp[:, :x, :]

    tmp = label.copy()
    label[:, :w-x, :] = tmp[:, x:, :]
    label[:, w-x:, :] = tmp[:, :x, :]

    return img, label, attention


def random_translation(img, label, attention):
    # t = [(0, 5), (5, 0), (-5, 0), (0, -5)]
    # idx = np.random.randint(0, len(t))
    # return translate(img, t[idx][0], t[idx][1]), translate(label, t[idx][0], t[idx][1]), \
    #        translate(attention, t[idx][0], t[idx][1])
    x = np.random.randint(-5, 5, size=1)
    y = np.random.randint(-5, 5, size=1)
    return translate(img, x, y), translate(label, x, y), translate(attention, x, y)


def horizontal_flip(img, label, attention):
    return np.fliplr(img), np.fliplr(label), np.fliplr(attention)


def vertical_flip(img, label, attention):
    return np.flipud(img), np.flipud(label), np.flipud(attention)


def add_gaussian_noise(img, label, attention):
    std = img.std()
    noisy = img + .0005 * np.random.normal(loc=0, scale=std, size=img.shape)
    return noisy, label, attention


def add_pepper_noise(img, label, attention):
    idx = np.random.randint(0, img.size, size=int(img.size * .05))
    org_shape = img.shape
    img = img.flatten()
    img[idx] = 0.
    return img.reshape(org_shape), label, attention


def add_salt_noise(img, label, attention):
    idx = np.random.randint(0, img.size, size=int(img.size * .05))
    org_shape = img.shape
    img = img.flatten()
    img[idx] = 255. if img.max() > 1. else 1.
    return img.reshape(org_shape), label, attention


def convert2pil(img):
    return Image.fromarray(img.astype("uint8"), mode="RGB")


def convert2np(pic):
    return np.array(pic.getdata()).reshape(pic.size[1], pic.size[0], 3).astype(np.float32)


def add_speckle_noise(img, label, attention):
    std = img.std()
    noisy = img + .0005 * img * np.random.normal(loc=0, scale=std, size=img.shape)
    return noisy, label, attention


def random_brightness(img, label, attention):
    b = ImageEnhance.Brightness(convert2pil(img))
    return convert2np(b.enhance(np.random.uniform(low=.7, high=1.))), label, attention


def random_contrast(img, label, attention):
    c = ImageEnhance.Contrast(convert2pil(img))
    return convert2np(c.enhance(np.random.uniform(low=.7, high=1.))), label, attention


def random_saturation(img, label, attention):
    c = ImageEnhance.Color(convert2pil(img))
    return convert2np(c.enhance(np.random.uniform(low=.7, high=1.))), label, attention


def histogram_equalization(img, label, attention):
    img_yuv = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGB2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB).astype(np.float32), label, attention


def skew(img, label, attention):
    warnings.warn("Skewing is not implemented yet!", RuntimeWarning)
    return img, label, attention


def random_rotation(img, label, attention):
    angle = np.random.randint(-20, high=20)   # 10
    return rotate(img, angle), rotate(label, angle), rotate(attention, angle)


def shear(img, label, attention):
    warnings.warn("Shearing is not implemented yet!", RuntimeWarning)
    return img, label, attention


def blur(img, label, attention):
    s = ImageEnhance.Sharpness(convert2pil(img))
    return convert2np(s.enhance(np.random.uniform(low=.7, high=1.))), label, attention


def random_erase(img, label, attention):
    img = img.copy()
    erase_size = 10
    max_val = 255 if img.max() > 1 else 1
    x = np.random.randint(erase_size, high=img.shape[0])
    img[x-erase_size:x, x-erase_size:x, :] = np.random.uniform(low=0, high=max_val, size=(erase_size, erase_size, 3))
    return img, label, attention


def add_brown_noise(img, label, attention):
    warnings.warn("Brownian noise is not implemented yet!", RuntimeWarning)
    return img, label, attention


if __name__ == "__main__":
    img = cv2.imread(
        "/media/digits2/9baf4857-aa15-462d-b5db-022aa5cc3dc5/MicrosurgicalTooltracking/data/EndoVis/labelled_train/img_000120_raw_train1.jpg"
    )
    label = np.load(
        "/media/digits2/9baf4857-aa15-462d-b5db-022aa5cc3dc5/MicrosurgicalTooltracking/data/EndoVis/training_labels/img_000120_raw_train1.npy"
    )
    img_out, _, _ = random_swap(img, label, None)
    cv2.imshow("out", img_out)
    cv2.waitKey(0)





