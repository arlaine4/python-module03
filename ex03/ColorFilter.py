import numpy as np


class ColorFilter:
    @classmethod
    def check_instance(cls, array):
        if not isinstance(array, np.ndarray):
            return False
        return True

    @classmethod
    def invert(cls, array):
        if not ColorFilter.check_instance(array):
            return None
        return 255 - array[:, :, :3]

    @classmethod
    def to_blue(cls, array):
        if not ColorFilter.check_instance(array):
            return None
        array[:, :, 0] = 0
        array[:, :, 1] = 0
        return array

    @classmethod
    def to_green(cls, array):
        if not ColorFilter.check_instance(array):
            return None
        array[:, :, 0] *= 0
        array[:, :, 2] *= 0
        return array

    @classmethod
    def to_red(cls, array):
        if not ColorFilter.check_instance(array):
            return None
        array[:, :, 1] -= array[:, :, 1]
        array[:, :, 2] -= array[:, :, 2]
        return array

    @classmethod
    def to_celluloid(cls, array):
        if not ColorFilter.check_instance(array):
            return None
        array[array < 64] = 0
        array[(array > 64) & (array < 128)] = 64
        array[array > 128] = 128
        return array

    @classmethod
    def to_grayscale(cls, array, filter, **kwargs):
        if not ColorFilter.check_instance(array) or type(filter) is not str:
            return None
        if filter in ['w', 'weighted']:
            try:
                channels = kwargs[list(kwargs.keys())[0]]
            except IndexError:
                return None
            if len(channels) != 3 or (channels[0] + channels[1] + channels[2] != 1):
                return None
            array = np.sum([array[:, :, 0] * channels[0], array[:, :, 1] * channels[1], array[:, :, 2] * channels[2]],
                           axis=0)
        elif filter in ['m', 'mean']:
            array[:, :, 0:3] = np.sum(array[:, :, 0:3] / 3, axis=2, keepdims=True)
        return array


