from ImageProcessor import *
from ColorFilter import *
import matplotlib.pyplot as plt
from copy import deepcopy

imp = ImageProcessor()
cf = ColorFilter()

arr = imp.load('42AI.png')
images = [cf.invert(arr), cf.to_blue(deepcopy(arr)), cf.to_green(deepcopy(arr)),
          cf.to_red(deepcopy(arr)), cf.to_celluloid(deepcopy(arr)),
          cf.to_grayscale(deepcopy(arr), 'w', channels=[0.2, 0.3, 0.5])]
fig = plt.figure(figsize=(8, 8))
for i, im in enumerate(images):
    fig.add_subplot(2, 3, i + 1)
    plt.imshow(im)
plt.show()
