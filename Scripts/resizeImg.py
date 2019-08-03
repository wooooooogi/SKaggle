import numpy as np
import os
import cv2
import time

data_path = "C:/Users/mabin/Dropbox/SKaggle/generative-dog-images/all-dogs/"
# data_path = "C:/Users/mabin/Dropbox/SKaggle/generative-dog-images/resized/"
data_name = os.listdir(data_path)
data_size = []
for i in data_name:
    size = os.stat(data_path + i).st_size
    data_size.append(size)

maxx = np.max(data_size)
maxxw = np.where(maxx == data_size)
minn = np.min(data_size)
tmpimg = cv2.imread(data_path + data_name[int(maxxw[0])])
img_height, img_width, img_channels = tmpimg.shape[:3]
if img_height >= img_width:
    img_size = img_height
else:
    img_size = img_width
img_size = 32
s1 = time.time()
for i in data_name:
    tmpimg = cv2.imread(data_path + i)
    tmpimg = cv2.resize(tmpimg, (img_size, img_size))
    cv2.imwrite(data_path + "../resized32/" + i, tmpimg)
    print(np.array(tmpimg).size)
print(time.time() - s1)