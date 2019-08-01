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
# save_file = "./model.ckpt"


#   Get path where dog image place
data_path = "C:/Users/mabin/Dropbox/SKaggle/generative-dog-images/resized/"
data_name = os.listdir(data_path)
data_size = []
for i in data_name:
    size = os.stat(data_path + i).st_size
    data_size.append(size)

img_size = 64
img_channels = 3

noise_size = 256    #   이게 image size와 관련이 있을거니... 함 잘 짜보자.
hidden_node1 = 3072   #   잘 고려해야지.
hidden_node2 = 768
hidden_node3 = 192
X = tf.compat.v1.placeholder(tf.float32, [None, img_size, img_size, img_channels])
Z = tf.compat.v1.placeholder(tf.float32, [None, noise_size])
keep_prob = tf.placeholder(tf.float32)

G_w1 = tf.Variable(tf.random.normal([256, 16*16*64], stddev=0.01))
G_b1 = tf.Variable(tf.random.normal([16*16*64], stddev=0.01))

G_w2 = tf.Variable(tf.random.normal([3, 3, 3, 64], stddev=0.01))

G_w3 = tf.Variable(tf.random.normal([3, 3, 3, 1], stddev=0.01))

G_w4 = tf.Variable(tf.random.normal([3, 3, 3, 1], stddev=0.01))

def generator(noise):
    G_L_1 = tf.nn.relu(tf.matmul(noise, G_w1) + G_b1)
    G_L_1 = tf.reshape(G_L_1, [-1, 8, 8, 64])

    G_L_2 = tf.nn.conv2d_transpose(value=G_L_1, filter=G_w2,
                                   output_shape=[-1, 16, 16, 3],
                                   strides=[1, 2, 2, 1], padding="SAME")
    G_L_3 = tf.nn.conv2d_transpose(value=G_L_2, filter=G_w3,
                                   output_shape=[-1, 32, 32, 3],
                                   strides=[1, 2, 2, 1], padding="SAME")
    output = tf.nn.conv2d_transpose(value=G_L_3, filter=G_w4,
                                   output_shape=[-1, 64, 64, 3],
                                   strides=[1, 2, 2, 1], padding="SAME")
    return output

D_w1 = tf.Variable(tf.random.normal([3, 3, 3, 32], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([32]))
# D_b1 = tf.Variable(tf.constant(0.1, [32]))

D_w2 = tf.Variable(tf.random.normal([3, 3, 32, 64], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([64]))
# D_b2 = tf.Variable(tf.constant(0.1, [64]))

D_w3 = tf.Variable(tf.random.normal([16*16*64, 256]))
D_b3 = tf.Variable(tf.zeros([256]))
# D_b3 = tf.Variable(tf.constant(0.1, [256]))

D_w4 = tf.Variable(tf.random.normal([256, 1], stddev=0.01))
D_b4 = tf.Variable(tf.random.normal([1], stddev=0.01))

def discriminator(input):
    #   64x64x1 -> 32x32x1x32
    D_L_1 = tf.nn.relu(tf.nn.conv2d(input, D_w1, strides=[1, 1, 1, 1], padding="SAME"))
    D_L_1 = tf.nn.max_pool(D_L_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    #   32x32x1x32 -> 16x16x1x64
    D_L_2 = tf.nn.relu(tf.nn.conv2d(D_L_1, D_w2, strides=[1, 1, 1, 1], padding="SAME"))
    D_L_2 = tf.nn.max_pool(D_L_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    #   16x16x1x64 -> 256
    D_L_3 = tf.reshape(D_L_2, [-1, 16*16*64])
    D_L_3 = tf.nn.relu(tf.matmul(D_L_3, D_w3) + D_b3)
    #   No dropout
    # D_L_3 = tf.nn.dropout(D_L_3, keep_prob=keep_prob)

    output = tf.nn.sigmoid(tf.matmul(D_L_3, D_w4) + D_b4)

    return output

G = generator(Z)
D_real = discriminator(X)
D_gene = discriminator(G)

loss_D = -tf.reduce_mean(tf.math.log(D_real) + tf.math.log(1 - D_gene))
loss_G = -tf.reduce_mean(tf.math.log(D_gene))

train_D = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0002).minimize(loss_D, var_list=[D_w1, D_b1, D_w2, D_b2, D_w3, D_b3, D_w4, D_b4])    #   compat.train.v1.
train_G = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0002).minimize(loss_G, var_list=[G_w1, G_b1, G_w2, G_w3, G_w4])

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables_initializer())  #  tf.train.Saver()
test_size = 10
image_save_friq = 10
# noise_test = np.random.normal(size=(test_size, noise_size))
epoch = 100
batch_size = 10
total_batch = int(len(data_size) / batch_size)
loss_val_D = 0
loss_val_G = 0
# batch = int(data_size / batch_size)
for i in range(epoch):#epoch
    for j in range(25):#total_batch
        #   Load data
        for k in range(batch_size):#batch_size
            k = j * batch_size
            if k % batch_size == 0:
                tmpimg = cv2.imread(data_path + data_name[k])   #, cv2.IMREAD_GRAYSCALE)
                tmpimg = tmpimg / 255.
                # tmpimg = tmpimg[:, :, np.newaxis]
                # tmpimg = tmpimg[np.newaxis, :, :, 1]
                tmpimg = tmpimg[np.newaxis, :, :, np.newaxis]
                imgdata = np.array(tmpimg)
            else:
                tmpimg = cv2.imread(data_path + data_name[k])
                tmpimg = tmpimg / 255.
                # tmpimg = tmpimg[:, :, np.newaxis]
                # tmpimg = tmpimg[np.newaxis, :, :, 1]
                tmpimg = tmpimg[np.newaxis, :, :, np.newaxis]
                imgdata = np.append(imgdata, tmpimg, axis=0)
        batch_xs = imgdata
        _noise = np.random.normal(size=(total_batch, noise_size))
        print(batch_xs)
        print(_noise)
        print(batch_xs.shape)
        print(_noise.shape)

        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: _noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: _noise})

        # saver.save(sess, "./model.ckpt")
    print("epoch {tryNum} finished".format(tryNum=i))
    if True:    #   i == 0 or (i + 1) % image_save_friq == 0:
        fig, ax = plt.subplots(1, test_size, figsize=(test_size, 1))
        for l in range(test_size):
            noise_test = np.random.normal(size=(test_size, noise_size))
            samples = sess.run(G, feed_dict={Z: noise_test})
            # samples = samples * 255.
            # print(samples)
            ax[l].set_axis_off()
            ax[l].imshow(np.reshape(samples[l], [img_size, img_size, 1]))
        plt.savefig('./Gen/{}.png'.format(str(i+1).zfill(3)), bbox_inches='tight')