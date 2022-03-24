import numpy as np


class ScrapBooker:
    @classmethod
    def crop(cls, array, dim, position=(0, 0)):
        if not isinstance(array, np.ndarray) or not isinstance(dim, tuple)\
                or not isinstance(position, tuple):
            return None
        if type(dim[0]) is not int or type(dim[1]) is not int or len(dim) != 2:
            return None
        if type(position[0]) is not int or type(position[1]) is not int or len(position) != 2:
            return None
        return array[position[0]:dim[0] + 1, position[1]:dim[1]]

    @classmethod
    def thin(cls, array, n, axis):
        if not isinstance(array, np.ndarray) or not isinstance(n, int) or\
                not isinstance(axis, int):
            return None
        if n <= 0 or not 0 <= axis <= 1:
            return None
        if axis == 0 and n >= array.shape[0] or (axis == 1 and n >= array.shape[1]):
            return None
        to_delete = [i - 1 for i in range(array.shape[axis] + 1) if i != 0 and i % 3 == 0]
        return np.delete(array, to_delete, axis)
