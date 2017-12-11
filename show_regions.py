import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from skimage.measure import label, regionprops


parent = "results/alexnet/384"
dirinfo = os.listdir(parent)
dirs = [os.path.join(parent, d) for d in dirinfo if os.path.isdir(os.path.join(parent, d))]

print("Found", len(dirs), "folders")
assert len(dirs) >= 18

target_shape = (224, 224)
images = np.zeros((len(dirs),) + target_shape + (3,), dtype=np.uint8)
masks = np.zeros((len(dirs),) + target_shape, dtype=np.uint8)

for i, d in enumerate(dirs):

    tmp = cv2.imread(os.path.join(d, 'sharp/cam.png'), 1)
    tmp = cv2.resize(tmp, target_shape)
    images[i] = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)

    tmp = cv2.imread(os.path.join(d, 'sharp/mask.png'), 1)
    tmp = cv2.resize(tmp, target_shape)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    _, masks[i] = cv2.threshold(tmp, 128, 255, cv2.THRESH_BINARY)
    tmp = cv2.cvtColor(masks[i], cv2.COLOR_GRAY2BGR)

    images[i] = images[i] * (tmp / 255)

for i in range(18):

    plt.figure(1)
    plt.subplot(3,6,i+1)
    label_img = label(masks[i])
    regions = regionprops(label_img)
    for props in regions:
        y0, x0 = props.centroid
        plt.plot(x0, y0, '.r', markersize=9)
    plt.imshow(masks[i])

    plt.figure(0)
    plt.subplot(3,6,i+1)
    for props in regions:
        y0, x0 = props.centroid
        plt.plot(x0, y0, '.r', markersize=9)
    plt.imshow(images[i])

#plt.tight_layout()
plt.show()
