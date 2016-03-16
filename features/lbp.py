# import the necessary packages
from skimage import feature
from skimage.color import rgb2gray
import numpy as np
from features.base import Descriptor


class LocalBinaryPatternsDescriptor(Descriptor):
    def __init__(self, num_points=9, radius=4, **kwargs):
        super(LocalBinaryPatternsDescriptor, self).__init__(**kwargs)
        self.num_points = num_points
        self.radius = radius
        self.eps = 1e-7

    def _calculate_descriptor(self, image):
        if len(image.shape) == 3:
            image = rgb2gray(image)
        lbp = feature.local_binary_pattern(image, self.num_points,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.num_points + 2),
                                 range=(0, self.num_points + 1))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + self.eps)

        # return the histogram of Local Binary Patterns
        return hist

    def size(self):
        return self.num_points + 1
