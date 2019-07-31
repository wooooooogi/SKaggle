import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import scipy
import time
import cv2


#   Remove previous weights and biases
tf.compat.v1.reset_default_graph()

#   Dir of model check point
save_file = "./model.ckpt"


#   Get path where dog image place
# data_path = "C:/Users/mabin/Dropbox/SKaggle/generative-dog-images/all-dogs/"
data_path = "C:/Users/mabin/Dropbox/SKaggle/generative-dog-images/resized/"
data_name = os.listdir(data_path)
data_size = []
for i in data_name:
    size = os.stat(data_path + i).st_size
    # size = os.path.getsize(data_path + i)
    data_size.append(size)
# maxx = np.max(data_size)
# maxxw = np.where(maxx == data_size)
# minn = np.min(data_size)
# tmpimg = cv2.imread(data_path + data_name[int(maxxw[0])])
# img_height, img_width, img_channels = tmpimg.shape[:3]
# if img_height >= img_width:
#     img_size = img_height
# else:
#     img_size = img_width
# Resize image data
# s1 = time.time()
# for i in data_name:
#     tmpimg = cv2.imread(data_path + i)
#     tmpimg = cv2.resize(tmpimg, (img_size, img_size))
#     cv2.imwrite(data_path + "../resized/" + i, tmpimg)
#     print(np.array(tmpimg).size)
# print(time.time() - s1)

img_size = 64
img_channels = 3

noise_size = 256    #   이게 image size와 관련이 있을거니... 함 잘 짜보자.
hidden_node = 512   #   잘 고려해야지.
X = tf.compat.v1.placeholder(tf.float32, [None, img_size * img_size * img_channels])
# Y = tf.compat.v1.placeholder(tf.float32, [None, img_size * img_size * img_channels])
Z = tf.compat.v1.placeholder(tf.float32, [None, noise_size])

G_w1 = tf.Variable(tf.random.normal([noise_size, hidden_node], stddev=0.01))
G_w2 = tf.Variable(tf.random.normal([hidden_node, img_size * img_size * img_channels], stddev=0.01))
G_b1 = tf.Variable(tf.random.normal([hidden_node]))
G_b2 = tf.Variable(tf.random.normal([img_size * img_size * img_channels]))

def generator(noise):
    hidden = tf.nn.relu(tf.matmul(noise, G_w1) + G_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, G_w2) + G_b2)
    return output

D_w1 = tf.Variable(tf.random.normal([img_size * img_size * img_channels, hidden_node], stddev=0.01))
D_w2 = tf.Variable(tf.random.normal([hidden_node, 1], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([hidden_node]))
D_b2 = tf.Variable(tf.zeros([1]))

def discriminator(input):
    hidden = tf.nn.relu(tf.matmul(input, D_w1) + D_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, D_w2) + D_b2)
    return output

G = generator(Z)

loss_D = -tf.reduce_mean(tf.math.log(discriminator(X)) + tf.math.log(1 - discriminator(G)))
loss_G = -tf.reduce_mean(tf.math.log(discriminator(G)))

train_D = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0002).minimize(loss_D, var_list=[D_w1, D_b1, D_w2, D_b2])    #   compat.train.v1.
train_G = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0002).minimize(loss_G, var_list=[G_w1, G_b1, G_w2, G_b2])

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

saver = tf.compat.v1.train.Saver()  #  tf.train.Saver()
noise_test = np.random.normal(size=(10, noise_size))
epoch = 100
batch_size = 100
total_batch = int(len(data_size) / batch_size)
# batch = int(data_size / batch_size)
for i in range(epoch):
    for j in range(total_batch):
        #   Load data
        for i in range(batch_size):
            i = j * batch_size
            if i % batch_size == 0:
                tmpimg = cv2.imread(data_path + data_name[i])
                tmpimg = np.array(tmpimg).ravel()
                tmpimg = tmpimg / 255.
                tmpimg = tmpimg[np.newaxis, :]
                imgdata = np.array(tmpimg)
            else:
                tmpimg = cv2.imread(data_path + data_name[i])
                tmpimg = np.array(tmpimg).ravel()
                tmpimg = tmpimg / 255.
                tmpimg = tmpimg[np.newaxis, :]
                imgdata = np.append(imgdata, tmpimg, axis=0)
        batch_xs = imgdata
        _noise = np.random.normal(size=(total_batch, noise_size))

        sess.run(train_D, feed_dict={X: batch_xs, Z: _noise})
        sess.run(train_G, feed_dict={Z: _noise})

        saver.save(sess, save_file)
    if epoch == 0 or (epoch + 1) % 10 == 0:
        samples = sess.run(G, feed_dict={Z: noise_test})

        fig, ax = plt.subplots(1, int(epoch/10), figsize=(int(epoch/10), 1))
        for i in range(10):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], [img_size, img_size, img_channels]))
        plt.savefig('./samples_ex_{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)




#   Restore weights and biases at check point
# saver.restore(sess, save_file)

# Deprecated code
# for i in data_name:
#     # size = os.stat(data_path + i).st_size
#     size = os.path.getsize(data_path + i)
#     img_size.append(size)
# img_width = np.size(tmpimg, 0)
# img_height = np.size(tmpimg, 1)
# img_channels = np.size(tmping, 2)