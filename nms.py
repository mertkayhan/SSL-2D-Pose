import numpy as np
from scipy.ndimage.filters import maximum_filter


def nms(heatmap):

    NMS_THRESH = [.2, .1, .1, .1, .1]

    for i in range(5):
        NMS_THRESH[i] = max(np.average(heatmap[:, :, i]) * 4., NMS_THRESH[i])
        NMS_THRESH[i] = min(NMS_THRESH[i], .3)

    window_size = 20
    peaks = [[], [], [], [], []]  # use only for data augmentation

    filtered = maximum_filter(heatmap, footprint=np.ones((window_size, window_size, 1)))
    suppressed = heatmap * np.equal(heatmap, filtered)
    suppressed = suppressed >= NMS_THRESH

    for ch in range(heatmap.shape[-1]):
        p = np.where(suppressed[:, :, ch] != 0)
        peaks[ch] += list(zip(p[1], p[0]))

    return peaks, suppressed

