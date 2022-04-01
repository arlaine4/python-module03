import numpy as np
from copy import deepcopy


class ScrapBooker:
    @classmethod
    def crop(cls, array, dim, position=(0, 0)):
        if not isinstance(array, np.ndarray) or not isinstance(dim, tuple)\
                or not isinstance(position, tuple):
            return None
        try:
            if type(dim[0]) is not int or type(dim[1]) is not int or len(dim) != 2:
                return None
            if type(position[0]) is not int or type(position[1]) is not int or len(position) != 2:
                return None
            if position[0] + dim[0] > array.shape[0] or position[1] + dim[1] > array.shape[1]:
                return None
        except (IndexError, TypeError):
            return None
        return array[position[0]:position[0] + dim[0], position[1]: position[1] + dim[1]]

    @classmethod
    def thin(cls, array, n, axis):
        if not isinstance(array, np.ndarray) or not isinstance(n, int) or\
                not isinstance(axis, int):
            return None
        if n <= 0 or not 0 <= axis <= 1:
            return None
        if len(array.shape) != 2 or axis == 0 and n >= array.shape[0] or (axis == 1 and n >= array.shape[1]):
            return None
        axis = 0 if axis == 1 else 1
        to_delete = [i - 1 for i in range(array.shape[axis] + 1) if i != 0 and i % n == 0]
        return np.delete(array, to_delete, axis)

    @classmethod
    def juxtapose(cls, array, n, axis):
        if not isinstance(array, np.ndarray) or not isinstance(n, int) or\
                not isinstance(axis, int):
            return None
        if axis not in [0, 1] or n <= 0:
            return None
        tmp_array = deepcopy(array)
        # axis = 0 if axis == 1 else 0
        for i in range(n - 1):
            array = np.concatenate((array, tmp_array), axis=axis)
        return array

    @classmethod
    def mosaic(cls, array, dim):
        if not isinstance(array, np.ndarray) or not isinstance(dim, tuple):
            return None
        for i in range(len(dim)):
            if i == 2:
                return None
            if type(dim[i]) is not int or dim[i] < 0:
                return None
        return np.array(np.tile(array, dim))
