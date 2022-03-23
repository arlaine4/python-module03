import numpy as np


class NumpyCreator:
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
        # add check for type tpl[0] and tpl[1]
        if not isinstance(tpl, tuple) or len(tpl) != 2:
            return None
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
        # add test for type shape[0] and shape[1]
        if not isinstance(shape, tuple) or type(value) not in \
                [int, float] or len(shape) != 2:
            return None
        array = np.full((shape[0], shape[1]), value)
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
        array = np.identity(n)
        return array
