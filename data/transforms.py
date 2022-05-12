from __future__ import absolute_import

import torchvision
from PIL import ImageOps, Image


from typing import Callable, Union, Tuple, Sequence

import numpy as np
import torchvision


class RandomProbChoice(torchvision.transforms.transforms.RandomTransforms):
    """Apply a randomly transformation chosen from a given set with some probability."""

    def __init__(self, transforms):
        # type: (Sequence[Union[Callable, Tuple[float, Callable]]]) -> None
        assert transforms, "You must specify at least one choice"

        callables = []
        self._probs = []
        for transformer in transforms:
            if isinstance(transformer, tuple):
                self._probs.append(transformer[0])
                callables.append(transformer[1])
            else:
                callables.append(transformer)
        if self._probs:
            assert len(self._probs) == len(callables)
        else:
            self._probs = None

        super(RandomProbChoice, self).__init__(callables)

    def __call__(self, x):
        t = np.random.choice(np.arange(len(self.transforms)), p=self._probs)
        return self.transforms[t](x)


class Identity(object):
    def __call__(self, x):
        return x

    def __repr__(self):
        return self.__class__.__name__ + "()"


Compose = torchvision.transforms.transforms.Compose
RandomApply = torchvision.transforms.transforms.RandomApply
RandomChoice = torchvision.transforms.transforms.RandomChoice

class Invert(object):
    """Invert the colors of a PIL image with the given probability."""

    def __call__(self, img):
        # type: (Image) -> Image
        return ImageOps.invert(img)

    def __repr__(self):
        return "vision.{}()".format(self.__class__.__name__)


class Convert(object):
    """Convert a PIL image to Greyscale, RGB or RGBA."""

    def __init__(self, mode):
        # type: (str) -> None
        assert mode in ("L", "RGB", "RGBA")
        self.mode = mode

    def __call__(self, img):
        # type: (Image) -> Image
        img = img.convert(self.mode)
        return img

    def __repr__(self):
        format_string = "vision." + self.__class__.__name__ + "("
        if self.mode is not None:
            format_string += "mode={}".format(self.mode)
        format_string += ")"
        return format_string


ToTensor = torchvision.transforms.transforms.ToTensor

class ConvertToNumpy(object):
    """Convert a PIL image to Greyscale, RGB or RGBA."""

    def __init__(self,):
        pass

    def __call__(self, img):
        # type: (Image) -> Image
        # print("size,", img.size)
        img =  np.array(img)
        # print("shape, ", img.shape)
        return img

    def __repr__(self):
        format_string = "vision." + self.__class__.__name__ + "("
        format_string += "ToNumpy"
        format_string += ")"
        return format_string


