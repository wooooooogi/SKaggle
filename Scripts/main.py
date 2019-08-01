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
hidden_node1 = 3072   #   잘 고려해야지.
hidden_node2 = 768
hidden_node3 = 192
X = tf.compat.v1.placeholder(tf.float32, [None, img_size * img_size * img_channels])
# Y = tf.compat.v1.placeholder(tf.float32, [None, img_size * img_size * img_channels])
Z = tf.compat.v1.placeholder(tf.float32, [None, noise_size])

G_w1 = tf.Variable(tf.random.normal([noise_size, hidden_node3], stddev=0.01))
G_b1 = tf.Variable(tf.random.normal([hidden_node3]))

G_w2 = tf.Variable(tf.random.normal([hidden_node3, hidden_node2], stddev=0.01))
G_b2 = tf.Variable(tf.random.normal([hidden_node2]))

G_w3 = tf.Variable(tf.random.normal([hidden_node2, hidden_node1], stddev=0.01))
G_b3 = tf.Variable(tf.random.normal([hidden_node1]))

G_S_w1 = tf.Variable(tf.random.normal([hidden_node1, img_size * img_size * img_channels], stddev=0.01))
G_S_b1 = tf.Variable(tf.random.normal([img_size * img_size * img_channels]))

def generator(noise):
    hidden1 = tf.nn.relu(tf.matmul(noise, G_w1) + G_b1)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, G_w2) + G_b2)
    hidden3 = tf.nn.relu(tf.matmul(hidden2, G_w3) + G_b3)
    output = tf.nn.sigmoid(tf.matmul(hidden3, G_S_w1) + G_S_b1)
    return output

D_w1 = tf.Variable(tf.random.normal([img_size * img_size * img_channels, hidden_node1], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([hidden_node1]))

D_w2 = tf.Variable(tf.random.normal([hidden_node1, hidden_node2], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([hidden_node2]))

D_w3 = tf.Variable(tf.random.normal([hidden_node2, hidden_node3], stddev=0.01))
D_b3 = tf.Variable(tf.zeros([hidden_node3]))

D_S_w1 = tf.Variable(tf.random.normal([hidden_node3, 1], stddev=0.01))
D_S_b1 = tf.Variable(tf.zeros([1]))

def discriminator(input):
    hidden1 = tf.nn.relu(tf.matmul(input, D_w1) + D_b1)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, D_w2) + D_b2)
    hidden3 = tf.nn.relu(tf.matmul(hidden2, D_w3) + D_b3)
    output = tf.nn.sigmoid(tf.matmul(hidden3, D_S_w1) + D_S_b1)
    return output

G = generator(Z)
D_real = discriminator(X)
D_gene = discriminator(G)

loss_D = -tf.reduce_mean(tf.math.log(D_real) + tf.math.log(1 - D_gene))
loss_G = -tf.reduce_mean(tf.math.log(D_gene))

#
train_D = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0002).minimize(loss_D, var_list=[D_w1, D_b1, D_w2, D_b2, D_w3, D_b3, D_S_w1, D_S_b1])    #   compat.train.v1.
train_G = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0002).minimize(loss_G, var_list=[G_w1, G_b1, G_w2, G_b2, G_w3, G_b3, G_S_w1, G_S_b1])

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
                tmpimg = cv2.imread(data_path + data_name[k])
                tmpimg = np.array(tmpimg).ravel()
                tmpimg = tmpimg / 255.
                tmpimg = tmpimg[np.newaxis, :]
                imgdata = np.array(tmpimg)
            else:
                tmpimg = cv2.imread(data_path + data_name[k])
                tmpimg = np.array(tmpimg).ravel()
                tmpimg = tmpimg / 255.
                tmpimg = tmpimg[np.newaxis, :]
                imgdata = np.append(imgdata, tmpimg, axis=0)
        batch_xs = imgdata
        _noise = np.random.normal(size=(total_batch, noise_size))
        # _noise = np.random.normal(total_batch, noise_size)
        # print(batch_xs)
        # print(_noise)
        # print(batch_xs.shape)
        # print(_noise.shape)

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
            ax[l].imshow(np.reshape(samples[l], [img_size, img_size, img_channels]))
        plt.savefig('./Gen/{}.png'.format(str(i+1).zfill(3)), bbox_inches='tight')
        # if i+1 == epoch:
        #     fig, ax = plt.subplots(image_save_friq, int(epoch / image_save_friq),
        #                            figsize=(int(epoch / image_save_friq), image_save_friq))
        #     for l in range(test_size):
        #         noise_test = np.random.normal(size=(test_size, noise_size))
        #         samples = sess.run(G, feed_dict={Z: noise_test})
        #         samples = samples * 255.
        #         print(samples)
        #         tmpl = int(l + i * image_save_friq)
        #         ax[tmpl, l].set_axis_off()
        #         ax[tmpl, l].imshow(np.reshape(samples[l], [img_size, img_size, img_channels]))
        #     plt.savefig('./Gen/All.png', bbox_inches='tight')
        #     plt.close(fig)




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