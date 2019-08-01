import cv2
import numpy as np
import os

data_path = "C:/Users/mabin/Dropbox/SKaggle/generative-dog-images/resized/"
data_name = os.listdir(data_path)
data_size = []
for i in data_name:
    size = os.stat(data_path + i).st_size
    data_size.append(size)

tmpimg = cv2.imread(data_path + data_name[80], cv2.IMREAD_GRAYSCALE)

tmpimg = tmpimg / 255.
tmpimg = tmpimg[np.newaxis, :, :]
print(tmpimg)