import matplotlib.pyplot as plt
import numpy as np
from PIL import *


class ImageProcessor:
    @classmethod
    def load(cls, path):
        try:
            img = Image.open(path)
        except (FileNotFoundError, UnidentifiedImageError, SyntaxError):
            print('File not found')
            return None
        pixels = np.array(img)
        print(f"Loading image of dimensions {pixels.shape[0]} x {pixels.shape[1]}")
        return pixels

    @classmethod
    def display(cls, array):
        try:
            plt.imshow(array)
            plt.show()
        except TypeError:
            print("Invalid RGB matrix")
            return None
