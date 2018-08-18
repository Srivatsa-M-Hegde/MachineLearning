
# coding: utf-8

# In[ ]:

from scipy import ndimage,misc
import scipy
import os
import matplotlib.pyplot as plt
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.python.client import device_lib
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import random as ran
import cv2
import numpy as np
import pandas as pd
#get_ipython().magic('matplotlib inline')



imagesize = 28


################################################################################33
#### Loading Data Set
#################################################################################
labelPath = 'G:/e-books/computer_science/Intro to ML/proj4/list_attr_celeba.txt'
labels = []
with open(labelPath,'r') as f:
    count = 0
    for line in f.readlines():
        if count == 0:
            numLabels = int(line.split('\n')[0])
        if count == 1:
            columns = line.split()
            labelIndex = columns.index('Eyeglasses')
        if count >= 2:
            labels.append(int(line.split()[labelIndex]))
        count += 1

labels = np.array(labels)       
unique, counts = np.unique(labels, return_counts=True)
labelCounts = dict(zip(unique, counts))
numNegative = labelCounts[-1]
numPositive = labelCounts[1]
numSamples = 5000
NegativeInds = np.argwhere(labels == -1).flatten()
PositiveInds = np.argwhere(labels == 1).flatten()
NegativeIndsDownsampled = np.reshape(np.random.choice(NegativeInds,numSamples),[-1,1])
NegativeLabels = np.reshape(labels[NegativeIndsDownsampled],[-1,1])
PositiveLabels = np.reshape(labels[PositiveInds],[-1,1])
numNegative = len(NegativeIndsDownsampled)
numPositive = len(PositiveInds)

path = "G:/e-books/computer_science/Intro to ML/proj4/img_align_celeba/img_align_celeba/"
dirContents = list(os.walk(path))
fnames = np.array([fname for fname in dirContents[0][2] if '.jpg' in fname]).flatten()
fnamesNegative = fnames[NegativeIndsDownsampled]
fnamesPositive = fnames[PositiveInds]
nImages = len(fnames)
NegativeFnames = list(fnames[NegativeIndsDownsampled].flatten())
PositiveFnames = list(fnames[PositiveInds].flatten())
NegativeImages = np.zeros([numNegative, imagesize,imagesize])
PositiveImages = np.zeros([numPositive, imagesize, imagesize])

def convert(image):
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

    
print("Beginning Load")
for i, fname in enumerate(NegativeFnames):
    fpath = os.path.join(path,fname)
    im = plt.imread(fpath)
    im_resize = scipy.misc.imresize(im,[imagesize,imagesize])
    im_resize = convert(im_resize)
    NegativeImages[i,:,:] = im_resize

for i, fname in enumerate(PositiveFnames):
    fpath = os.path.join(path,fname)
    #Read Image
    im = plt.imread(fpath)
    #Append to array
    im_resize = scipy.misc.imresize(im,[imagesize,imagesize])
    im_resize = convert(im_resize)
    PositiveImages[i,:,:] = im_resize
    
PositiveImages = np.array(PositiveImages)
NegativeImages = np.array(NegativeImages)
all_x = np.concatenate((NegativeImages, PositiveImages), axis = 0).astype('float32') / 255.0
all_y = np.concatenate((NegativeLabels,PositiveLabels),axis = 0)

#########################################################################
##Splitting Data into Test and Train
#########################################################################

X_train, X_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.33, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
X_train = np.reshape(X_train,[-1,imagesize*imagesize])
X_test = np.reshape(X_test,[-1,imagesize*imagesize])
X_valid = np.reshape(X_valid,[-1,imagesize*imagesize])
y_train = y_train.flatten()
y_test = y_test.flatten()
y_valid = y_valid.flatten()

print("End of load")

######################################################################3
###One hot Vectorization
##########################################################################
y_train_label = []

for i,label in enumerate(y_train):
    if label == -1:
        y_train_label.append([1,0])
    elif label == 1:
        y_train_label.append([0,1])
y_train_label = np.array(y_train_label)        
y_test_label = []

for i,label in enumerate(y_test):
    if label == -1:
        y_test_label.append([1,0])
    elif label == 1:
        y_test_label.append([0,1])
y_test_label = np.array(y_test_label)

y_valid_label = []

for i,label in enumerate(y_valid):
    if label == -1:
        y_valid_label.append([1,0])
    elif label == 1:
        y_valid_label.append([0,1])
y_valid_label = np.array(y_valid_label)

x = tf.placeholder(tf.float32, [None, imagesize*imagesize])
W = tf.Variable(tf.zeros([imagesize*imagesize, 2]))
b = tf.Variable(tf.zeros([2]))

y_ = tf.placeholder(tf.float32, [None, 2])
#########################################################################
#######Function Definition
#########################################################################
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def return_batch(X, y, lower, upper):
    return X[lower:upper], y[lower:upper]

############################################################################
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x,[-1,imagesize,imagesize,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


epochs = 500
batch_size = 50


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    lower = 0
    upper = batch_size
    
    for i in range(epochs):
   
        
   
        batch_x,batch_y = return_batch(X_train,y_train_label,lower,upper)

        lower += batch_size
        upper += batch_size
        if upper >= len(X_train):
            lower = 0
            upper = batch_size
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: X_train, y_: y_train_label, keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        _,c = sess.run([train_step, cross_entropy], feed_dict={x: batch_x, y_: batch_y, keep_prob:0.7})
      
        train_accuracy = accuracy.eval(feed_dict = {x:X_train, y_: y_train_label, keep_prob: 0.8})
        valid_accuracy = accuracy.eval(feed_dict = {x:X_valid, y_: y_valid_label, keep_prob: 0.8})
        test_accuracy = accuracy.eval(feed_dict = {x:X_test, y_: y_test_label, keep_prob: 0.8})
 
print(valid_accuracy)
print(test_accuracy)


# In[ ]:



