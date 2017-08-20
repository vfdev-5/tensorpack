#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: _test.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import logging
import unittest

from tensorpack.dataflow.imgaug.misc import Resize
from tensorpack.dataflow.imgaug import deserialize
from tensorpack.dataflow.imgaug.base import serialize


class TestImgAug(unittest.TestCase):

    def test_serialization(self):
        resize_aug = Resize(shape=(128, 128))
        resize_aug_serialized = serialize(resize_aug)
        self.assertTrue(isinstance(resize_aug_serialized, dict))
        resize_aug_deserialized = deserialize(resize_aug_serialized)
        self.assertTrue(isinstance(resize_aug_deserialized, Resize))
        self.assertEqual(resize_aug_deserialized.get_config(), resize_aug.get_config(), "Configs are different")


def run_test_case(case):
    suite = unittest.TestLoader().loadTestsFromTestCase(case)
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    from tensorpack.utils import logger
    logger.setLevel(logging.CRITICAL)
    run_test_case(TestImgAug)



## OLD STUFF

# import sys
# import cv2
# from . import AugmentorList
# from .crop import *
# from .imgproc import *
# from .noname import *
# from .deform import *
# from .noise import SaltPepperNoise


# anchors = [(0.2, 0.2), (0.7, 0.2), (0.8, 0.8), (0.5, 0.5), (0.2, 0.5)]
# augmentors = AugmentorList([
#     Contrast((0.8, 1.2)),
#     Flip(horiz=True),
#     GaussianDeform(anchors, (360, 480), 0.2, randrange=20),
#     # RandomCropRandomShape(0.3),
#     SaltPepperNoise()
# ])
#
# img = cv2.imread(sys.argv[1])
# newimg, prms = augmentors._augment_return_params(img)
# cv2.imshow(" ", newimg.astype('uint8'))
# cv2.waitKey()
#
# newimg = augmentors._augment(img, prms)
# cv2.imshow(" ", newimg.astype('uint8'))
# cv2.waitKey()
