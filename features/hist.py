from skimage.color import rgb2gray
import numpy as np

from base import Descriptor


class HistDescriptor(Descriptor):
    def __init__(self, num_points=20, **kwargs):
        super(HistDescriptor, self).__init__(**kwargs)
        self.num_points = num_points
        self.eps = 1e-7

    def _calculate_descriptor(self, img):
        if len(img.shape) == 3:
            img = rgb2gray(img)

        bins = [x * 1.0 / (self.num_points + 2) for x in np.arange(0, self.num_points + 2)]
        (hist, _) = np.histogram(img.ravel(), bins=bins,
                                 range=(0, self.num_points + 1))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + self.eps)

        # return the histogram of Local Binary Patterns
        return hist

    def size(self):
        return self.num_points + 1
