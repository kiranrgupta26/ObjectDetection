# -*- coding: utf-8 -*-
"""
Created on Jul 29 07:50:23 2020

@author: kiran
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import pandas as pd
import math
import cv2
from imgaug import augmenters as iaa
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import csv
import tensorflow as tf
import tensorflow.keras as keras

dataset = pd.read_csv('Downloads/WheatHEad Detection/train.csv',header=None)

image_data = dataset.iloc[1:,0].values #to get all image names
indexes = np.unique(image_data,return_index=True)[1]# to get unique image name index
unique_image_name =  [image_data[index] for index in sorted(indexes)]

print(len(unique_image_name))

m = len(unique_image_name)
#m=100
x_train = np.zeros((m,608,608),dtype=np.float32)

for i in range(m):   
    file_name = 'Downloads/WheatHEad Detection/train/'+unique_image_name[i] +'.jpg'
    img = mpimg.imread(file_name)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.resize(img,(608,608))
    img = img/255
    x_train[i] = img
    
plt.imshow(x_train[99])
print(x_train.shape)

height_of_grid = 608/19
width_of_grid = 608/19
y_train = np.zeros((m,19,19,5),dtype=np.float32)


#bbox - a bounding box, formatted as a Python-style list of [xmin, ymin, width, height]
for i in range(m):   
    image_id = pd.DataFrame(np.array(dataset[0][1:]==unique_image_name[i]))
    image_id_index =image_id.index[image_id[0]==True].tolist()
    for j in (image_id_index):      
        image = dataset[3][j+1]
        split_data = np.array([image[1:len(image)-1].split(",")])
        x_min = (float(split_data[0][0]) * 608)/1024
        y_min = (float(split_data[0][1]) * 608)/1024
        grid_col = math.ceil(float(x_min)/width_of_grid)
        grid_row = math.ceil(float(y_min)/height_of_grid)
        
        x_min_ratio_per_grid = float(x_min)/width_of_grid
        y_min_ratio_per_grid = float(y_min)/height_of_grid
        
        y_train[i][grid_row-1][grid_col-1][0] = 1
           
        y_train[i][grid_row-1][grid_col-1][1] = float(x_min_ratio_per_grid- int(x_min_ratio_per_grid)) 
        
        y_train[i][grid_row-1][grid_col-1][2] = float(y_min_ratio_per_grid - int(y_min_ratio_per_grid))
        
        y_train[i][grid_row-1][grid_col-1][3] = ((float(split_data[0][2])*608)/1024)/width_of_grid
        y_train[i][grid_row-1][grid_col-1][4] = ((float(split_data[0][3])*608)/1024)/height_of_grid
      

#Draw Bounding Box
def convert_boxes_to_xywh(y_train):    
    boxes=list()
    for i in range(19):
        for j in range(19):
            if y_train[0][i][j][0]>=0.6:
                xywh = list()       
                xywh.append((y_train[0][i][j][1]+j)*32)
                xywh.append((y_train[0][i][j][2]+i)*32)
                xywh.append((y_train[0][i][j][3])*32)
                xywh.append((y_train[0][i][j][4])*32)
                boxes.append(xywh)
    return boxes

#19,19,5
def calculate_loss(y_true,y_pred):
    
    coord =(tf.square(y_true[:,:,:,1] - y_pred[:,:,:,1]) + tf.square(y_true[:,:,:,2] - y_pred[:,:,:,2]))
    wh =  (tf.square(tf.sqrt(y_true[:,:,:,3]) - tf.sqrt(y_pred[:,:,:,3])) + tf.square(tf.sqrt(y_true[:,:,:,4]) - tf.sqrt(y_pred[:,:,:,4])))
    prob = (tf.square(y_true[:,:,:,0] - y_pred[:,:,:,0]))
    
    loss = (prob+coord+wh)
    #loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))
    return loss


model = keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(608,608,1),name="input1"))
model.add(tf.keras.layers.Conv2D(8,3,padding='same',strides=(1,1),name='conv1'))
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same',strides=2,name='pool1'))
model.add(tf.keras.layers.Conv2D(16,3,padding='same',strides=(1,1),name='conv2'))
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same',strides=2,name='pool2'))
model.add(tf.keras.layers.Conv2D(64,3,padding='same',strides=(1,1),name='conv3'))
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.Conv2D(64,3,padding='same',strides=(1,1),name='conv4'))
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.Conv2D(128,3,padding='same',strides=(1,1),name='conv5'))
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same',strides=2,name='pool3'))
model.add(tf.keras.layers.Conv2D(256,3,padding='same',strides=(1,1),name='conv6'))
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.Conv2D(128,3,padding='same',strides=(1,1),name='conv7'))
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.Conv2D(256,3,padding='same',strides=(1,1),name='conv8'))
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same',strides=2,name='pool4'))
model.add(tf.keras.layers.Conv2D(256,3,padding='same',strides=(1,1),name='conv9'))
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.Conv2D(128,3,padding='same',strides=(1,1),name='conv10'))
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.Conv2D(64,3,padding='same',strides=(1,1),name='conv11'))
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.Conv2D(64,3,padding='same',strides=(1,1),name='conv12'))
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.Conv2D(16,3,padding='same',strides=(1,1),name='conv13'))
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same',strides=2,name='pool5'))
model.add(tf.keras.layers.Conv2D(5,3,padding='same',strides=(1,1),name='conv14'))
model.add(tf.keras.layers.ReLU())
model.summary()


optimizer = tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.9)

xtrain = x_train.reshape((m,608,608,1))

model.compile(optimizer=optimizer,loss=calculate_loss,metrics=['accuracy'])
history = model.fit(xtrain,y_train,batch_size=16,epochs=127)

model.save('wheat_head_detection.h5')

#Testing
file_name = 'Downloads/WheatHEad Detection/train/c64a4006c.jpg'
img = mpimg.imread(file_name)
img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img = cv2.GaussianBlur(img,(3,3),0)
img = cv2.resize(img,(608,608))
img = img/255
test_501 = img

test_501 = test_501.reshape((1,608,608,1))
print(test_501.shape)

ypred = model.predict(test_501)
boxes = convert_boxes_to_xywh(ypred)

file_name = 'Downloads/WheatHEad Detection/train/c64a4006c.jpg'
im = Image.open(file_name)
im = im.resize((608,608))
figure,axes = plt.subplots(1)
axes.imshow(im)

image_id = '53f253011'
rows = []

file_name = 'Downloads/WheatHEad Detection/test/'+image_id +'.jpg'
im = Image.open(file_name)
#im = im.resize((608,608))
figure,axes = plt.subplots(1)
axes.imshow(im)
org_boxes=[]
for i in range(len(boxes)):
    box = boxes[i]
    X = box[0]*1024/608
    Y = box[1]*1024/608
    org_height = (box[2]*1024)/608
    org_width = (box[3]*1024)/608
    row = []
    prediction=[]
    prediction.append(1.0)
    prediction.append(X)
    prediction.append(Y)
    prediction.append(org_height)
    prediction.append(org_width)
    row.append(image_id)
    row.append(prediction)
    rows.append(row)
    rect = patches.Rectangle((X, Y), org_height,org_width,linewidth=1,edgecolor='r',facecolor='none')
    axes.add_patch(rect)

plt.show()