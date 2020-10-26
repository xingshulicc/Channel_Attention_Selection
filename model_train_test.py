# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Fri Jun 12 10:20:58 2020

@author: Admin
"""

import os
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from keras import backend as K

from keras.callbacks import ModelCheckpoint

#from Bridge_VGG19 import Bridge_VGG
#from VGG_model import VGG_16
#from ResNet_50_test import ResNet50
#from Weakly_Dense_Net import Weakly_DenseNet
#from SE_weakly_densenet import Weakly_DenseNet
from keras.applications.vgg19 import VGG19

from learning_rate import choose

#pre-parameters
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # '1' or '0' GPU

img_height, img_width = 224, 224

if K.image_dim_ordering() == 'th':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
input_tensor = Input(shape=(img_width, img_height, 3))


batch_size = 16
epochs = 500

train_data_dir = os.path.join(os.getcwd(), 'image_Data/train')
validation_data_dir = os.path.join(os.getcwd(), 'image_Data/validation')

num_classes = 24
nb_train_samples = 10402
nb_validation_samples = 2159

#model = VGG_16(input_shape = input_shape, classes = num_classes)


base_model = VGG19(include_top = False, weights = 'imagenet', 
                   input_tensor = input_tensor, pooling = 'avg')
x = base_model.output
x = Dense(512, activation = 'relu')(x)
x = Dropout(rate = 0.5)(x)
x = Dense(num_classes, activation = 'softmax')(x)
model = Model(base_model.input, outputs = x, name = 'VGG19_ImageNet_Pretrain')
for layer in model.layers:
    layer.trainable = True

##print layers' index and name-- check weakly densenet layer name and index
#for i, layer in enumerate(model.layers):
#    print(i, layer.name)

##Determine how many layers we should freeze, i.e we will freeze the first
##5 layers and fine-tune the rest-- we will set weakly densenet pre-trained layers 
#for layer in model.layers[:6]:
#    layer.trainable = False
#for layer in model.layers[6:]:
#    layer.trainable = True


optimizer = SGD(lr = 0.001, momentum = 0.9, nesterov = True) 
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
model.summary()


train_datagen = ImageDataGenerator(rescale = 1. / 255, 
                                   rotation_range = 30, 
                                   width_shift_range = 0.2, 
                                   height_shift_range = 0.2, 
                                   horizontal_flip = True, 
                                   zoom_range = 0.2, 
                                   fill_mode = 'nearest')

test_datagen = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical')


#set learning rate schedule
lr_monitorable = True
lr_reduce = choose(lr_monitorable = lr_monitorable)

#save best model weights based on validation accuracy
save_dir = os.path.join(os.getcwd(), 'VGG_crop_pest_disease_train')
weights_name = 'keras_best_weights.h5'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, weights_name)

checkpoint = ModelCheckpoint(filepath = save_path, 
                             monitor = 'val_acc', 
                             verbose = 1, 
                             save_best_only = True, 
                             mode = max, 
                             save_weights_only = True, 
                             period = 1)

#set callbacks for model fit
callbacks = [lr_reduce, checkpoint]

#model fit
hist = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples //batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples //batch_size, 
    callbacks=callbacks)

#print acc and stored into acc.txt
f = open('/home/xingshuli/Desktop/acc.txt','w')
f.write(str(hist.history['acc']))
f.close()
#print val_acc and stored into val_acc.txt
f = open('/home/xingshuli/Desktop/val_acc.txt','w')
f.write(str(hist.history['val_acc']))
f.close()
#print val_loss and stored into val_loss.txt   
f = open('/home/xingshuli/Desktop/val_loss.txt', 'w')
f.write(str(hist.history['val_loss']))
f.close()

#the reasonable accuracy of model should be calculated based on
#the value of patience in EarlyStopping: accur = accur[-patience + 1:]/patience
Er_patience = 10
accur = []
with open('/home/xingshuli/Desktop/val_acc.txt','r') as f1:
    data1 = f1.readlines()
    for line in data1:
        odom = line.strip('[]\n').split(',')
        num_float = list(map(float, odom))
        accur.append(num_float)
f1.close()

y = sum(accur, [])
ave = sum(y[-Er_patience:]) / len(y[-Er_patience:])
print('Validation Accuracy = %.4f' % (ave))


