from skimage.color import rgb2gray
import numpy as np

from base import Descriptor


class HistDescriptor(Descriptor):
    def __init__(self, num_points=32, **kwargs):
        super(HistDescriptor, self).__init__(**kwargs)
        self.num_points = num_points
        self.eps = 1e-7

    def _calculate_descriptor(self, img, *args, **kwargs):
        if len(img.shape) == 3:
            img = rgb2gray(img)
        # if 'mean' in kwargs:
        #    mean = kwargs['mean']
        #    img2 = img / mean * 128
        # else:
        img2 = img
        bins = [x * 1.0 / (self.num_points + 2) for x in np.arange(0, self.num_points + 2)]
        # (hist, _) = np.histogram(img.ravel(), bins=bins,
        #                         range=(0, self.num_points + 1))
        (hist2, _) = np.histogram(img2.ravel(), bins=bins,
                                  range=(0, self.num_points + 1 - 5))
        # normalize the histogram
        hist = hist2.astype(np.float)
        # hist = hist / sum(hist)
        # hist /= (hist.sum() + self.eps)

        statistics = np.asarray(
            [])  # np.asarray([float(img.min()), float(img.max()), img.mean(), img.std(), img.var()])
        # return the histogram of Local Binary Patterns
        return np.concatenate((statistics, hist))

    def size(self):
        return self.num_points + 1


def hist_distance(h1, h2, power_=2):
    #  print h1, h2
    needed, availiable = map(list, zip(*[(y - x if x >= y else 0, y - x if y >= x else 0) for x, y in zip(h1, h2)]))
    # print needed, availiable
    dist = 0
    j = 0
    for i in xrange(len(needed)):
        val = needed[i]
        while val != 0:
            if availiable[j] == 0:
                j += 1
                continue
            if availiable[j] >= abs(val):
                dist += abs(val) * abs(pow(i - j, power_))
                availiable[j] += val
                val = 0
            else:
                dist += abs(availiable[j]) * abs(pow(i - j, power_))
                # print availiable
                val += availiable[j]
                availiable[j] = 0

    return dist
