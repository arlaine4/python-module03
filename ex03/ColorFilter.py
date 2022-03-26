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
        array[:, :, 0] = 0
        array[:, :, 2] = 0
        return array

    @classmethod
    def to_red(cls, array):
        if not ColorFilter.check_instance(array):
            return None
        array[:, :, 1] = 0
        array[:, :, 2] = 0
        return array

    @classmethod
    def to_celluloid(cls, array):
        if not ColorFilter.check_instance(array):
            return None
        array[array < 64] = 0
        array[(array > 64) & (array < 128)] = 64
        array[array > 128] = 128
        """array[array < 51] = 0
        array[(array > 51) & (array < 102)] = 51
        array[(array > 102) & (array < 153)] = 102
        array[(array > 153) & (array < 204)] = 153
        array[(array > 204) & (array < 255)] = 204
        array[array == 255] = 255"""
        return array

    @classmethod
    def to_grayscale(cls, array):
        if not ColorFilter.check_instance(array):
            return None
        return array

