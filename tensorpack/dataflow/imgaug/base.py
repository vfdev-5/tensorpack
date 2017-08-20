# -*- coding: UTF-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

from abc import abstractmethod, ABCMeta
from ...utils.utils import get_rng
import six
from six.moves import zip

__all__ = ['Augmentor', 'ImageAugmentor', 'AugmentorList', 'serialize']


@six.add_metaclass(ABCMeta)
class Augmentor(object):
    """ Base class for an augmentor"""

    def __init__(self):
        self.reset_state()

    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != 'self':
                    setattr(self, k, v)

    def reset_state(self):
        """ reset rng and other state """
        self.rng = get_rng(self)

    def augment(self, d):
        """
        Perform augmentation on the data.
        """
        d, params = self._augment_return_params(d)
        return d

    def _augment_return_params(self, d):
        """
        Augment the image and return both image and params
        """
        prms = self._get_augment_params(d)
        return (self._augment(d, prms), prms)

    @abstractmethod
    def _augment(self, d, param):
        """
        Augment with the given param and return the new data.
        The augmentor is allowed to modify data in-place.
        """

    def _get_augment_params(self, d):
        """
        Get the augmentor parameters.
        """
        return None

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return self.rng.uniform(low, high, size)

    def get_config(self):
        """Get the config of the augmentor.

        Augmentor's config is a Python dictionary (serializable)
        containing the configuration of the augmentor.
        """
        config = {}
        for k, v in self.__dict__.items():
            if not (callable(v) or k == 'rng'):
                config[k] = v
        return config

    @classmethod
    def from_config(cls, config):
        """Creates an augmentor from its config.

        This method is the reverse of `get_config`,
        capable of instantiating the same augmentor from the config
        dictionary.

        Args:
            config: A Python dictionary, typically the output of get_config.

        Returns:
            augmentor instance.
        """
        return cls(**config)

    def __repr__(self):
        return str(serialize(self))

    def __str__(self):
        return self.__repr__()


class ImageAugmentor(Augmentor):
    def _augment_coords(self, coords, param):
        """
        Augment the coordinates given the param.
        By default, keeps coordinates unchanged.
        If a subclass changes coordinates but couldn't implement this method,
        it should ``raise NotImplementedError()``.

        Args:
            coords: Nx2 floating point nparray where each row is (x, y)
        Returns:
            new coords
        """
        return coords


class AugmentorList(ImageAugmentor):
    """
    Augment by a list of augmentors
    """

    def __init__(self, augmentors):
        """
        Args:
            augmentors (list): list of :class:`ImageAugmentor` instance to be applied.
        """
        self.augmentors = augmentors
        super(AugmentorList, self).__init__()

    def _get_augment_params(self, img):
        # the next augmentor requires the previous one to finish
        raise RuntimeError("Cannot simply get all parameters of a AugmentorList without running the augmentation!")

    def _augment_return_params(self, img):
        assert img.ndim in [2, 3], img.ndim

        prms = []
        for a in self.augmentors:
            img, prm = a._augment_return_params(img)
            prms.append(prm)
        return img, prms

    def _augment(self, img, param):
        assert img.ndim in [2, 3], img.ndim
        for aug, prm in zip(self.augmentors, param):
            img = aug._augment(img, prm)
        return img

    def _augment_coords(self, coords, param):
        for aug, prm in zip(self.augmentors, param):
            coords = aug._augment_coords(coords, prm)
        return coords

    def reset_state(self):
        """ Will reset state of each augmentor """
        for a in self.augmentors:
            a.reset_state()


def serialize(augmentor):
    """Serialize an augmentor.

    Args:
        augmentor: an Augmentor object.

    Returns:
        dictionary with config.
    """
    return {'class_name': augmentor.__class__.__name__,
            'config': augmentor.get_config()}
