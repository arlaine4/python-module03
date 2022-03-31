import numpy as np


class NumPyCreator:
    @classmethod
    def from_list(cls, lst):
        if not isinstance(lst, list):
            return None
        if len(lst) > 1 and type(lst[0]) is list:
            base_sub_len = len(lst[0])
            for sub_lst in lst:
                if len(sub_lst) != base_sub_len:
                    return None
        array = np.array(lst)
        return array

    @classmethod
    def from_tuple(cls, tpl):
        if not isinstance(tpl, tuple):
            return None
        for i in range(len(tpl)):
            try:
                if len(tpl[0]) != len(tpl[i]):
                    return None
            except TypeError:
                break
        array = np.array(tpl)
        return array

    @classmethod
    def from_iterable(cls, itr):
        if not hasattr(itr, '__iter__'):
            return None
        array = np.fromiter(itr, int)
        return array

    @classmethod
    def from_shape(cls, shape, value=0):
        if not isinstance(shape, tuple) or type(value) not in \
                [int, float] or len(shape) != 2:
            return None
        for elem in shape:
            if type(elem) is not int or elem < 0:
                return None
        array = np.full((shape[0], shape[1]), value, dtype='float64')
        return array

    @classmethod
    def random(cls, shape):
        if not isinstance(shape, tuple):
            return None
        array = np.random.random_sample((shape[0], shape[1]))
        return array

    @classmethod
    def identity(cls, n):
        if not isinstance(n, int):
            return None
        if n < 0:
            return None
        array = np.identity(n)
        return array
