# -*- coding: UTF-8 -*-
# File: __init__.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import os
from pkgutil import iter_modules

__all__ = ['deserialize']


def global_import(name):
    p = __import__(name, globals(), locals(), level=1)
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    del globals()[name]
    for k in lst:
        globals()[k] = p.__dict__[k]
        __all__.append(k)


try:
    import cv2  # noqa
except ImportError:
    from ...utils import logger
    logger.warn("Cannot import 'cv2', therefore image augmentation is not available.")
else:
    _CURR_DIR = os.path.dirname(__file__)
    for _, module_name, _ in iter_modules(
            [os.path.dirname(__file__)]):
        srcpath = os.path.join(_CURR_DIR, module_name + '.py')
        if not os.path.isfile(srcpath):
            continue
        if not module_name.startswith('_'):
            global_import(module_name)


def deserialize(config):
    """Instantiate a layer from a config dictionary.

    Args:
        config: dict of the form {'class_name': str, 'config': dict}

    Returns:
        Augmentor instance
    """
    assert isinstance(config, dict), "Argument config should be of the form {'class_name': str, 'config': dict}"
    if 'class_name' not in config or 'config' not in config:
        raise ValueError('Improper config format: ' + str(config))

    global_classes = globals()
    class_name = config['class_name']
    cls = global_classes[class_name]

    if cls is None:
        raise ValueError('Unknown class: %s' % class_name)
    if not hasattr(cls, 'from_config'):
        raise ValueError('Class %s is not derived from Augmentor class' % class_name)

    return cls.from_config(config['config'])
