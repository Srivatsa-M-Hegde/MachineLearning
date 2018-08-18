
# coding: utf-8

# In[7]:

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
import os
import cv2
import numpy as np
count = 0
count1 = 0
#import Image
from scipy.misc import imresize
path_train = "./Numerals/"
train = []
train1 = []
train_label = []
test = []
test_label = []
for _,dirs,_ in os.walk(path_train):
    for files in dirs:
        directory = path_train + files + '/'
        for fname in os.listdir(directory):
           # print(os.listdir(directory))
            if(fname.endswith(".png")):
                image = cv2.imread(directory+fname)
                im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                im_resize = cv2.resize(im_gray, (28, 28))
                train.append(np.reshape(im_resize, (1, 784)))
                train_label.append(count)
        count = count + 1
        
#train_label = np.array(train_label)

train_label_new = [] 
import matplotlib.pyplot as plt
for var in train_label :
    if (var == 0) :
         train_label_new.append([0,0,0,0,0,0,0,0,0,0])
    elif (var == 1) :
         train_label_new.append([0,1,0,0,0,0,0,0,0,0])
    elif (var == 2) :
          train_label_new.append([0,0,1,0,0,0,0,0,0,0])
    elif (var == 3) :
          train_label_new.append([0,0,0,1,0,0,0,0,0,0])
    elif (var == 4) :
         train_label_new.append([0,0,0,0,1,0,0,0,0,0])
    elif (var == 5) :
         train_label_new.append([0,0,0,0,0,1,0,0,0,0])
    elif (var == 6) :
         train_label_new.append([0,0,0,0,0,0,1,0,0,0])
    elif (var == 7) :
         train_label_new.append([0,0,0,0,0,0,0,1,0,0])
    elif (var == 8) :
         train_label_new.append([0,0,0,0,0,0,0,0,1,0])
    elif (var == 9) :
         train_label_new.append([0,0,0,0,0,0,0,0,0,1])

train = np.asarray(train)
for x in range (0, 999):
    train[x] = (255 - train[x])/255.
#train = (255 - train)/255.
#train = np.reshape(train,[-1,28,28,1])
train = np.reshape(train, [train.shape[0], train.shape[2]])
train_label = np.asarray(train_label_new)
#• Input
x = tf.placeholder(tf.float32, [None, 784])
#• Parameter Variable
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#• Softmax regression model:
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(
-tf.reduce_sum(y_ * tf.log(y),
reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()
logistic_accuracy = []
logistic_loss = []
for _ in range(100):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    _,c= sess.run([train_step,cross_entropy], feed_dict={x: batch_xs, y_:batch_ys})
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    print("MNIST test data")
    print(sess.run(accuracy, feed_dict={x:
    mnist.test.images, y_: mnist.test.labels}))
    print("USPS test data")
    print(sess.run(accuracy, feed_dict={x:
    train[0:999, ], y_: train_label[0:999,]}))
    logistic_accuracy.append(sess.run(accuracy, feed_dict={x:train[0:999, ], y_: train_label[0:999,]}))
    

