# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Fri Oct 23 13:31:34 2020

@author: Admin
"""
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Concatenate
from keras.layers import add
from keras.layers import Dense
from keras.layers import Dropout

from keras import backend as K
from keras.models import Model

from squeeze_excitation_layer import squeeze_excite_block
from keras.utils import plot_model

if K.image_data_format() == 'channels_first':
    bn_axis = 1
else:
    bn_axis = -1

def _select_kernel(inputs, kernels, filters, ratio, block):
    '''
    kernels = [3, 5]
    ratio = 4
 
    '''
    base_name = 'sk_block_' + str(block) + '_'
    k_1, k_2 = kernels
    
    x_1 = Conv2D(filters = filters, 
                 kernel_size = k_1, 
                 strides = (1, 1), 
                 padding = 'same', 
                 name = base_name + 'conv_1')(inputs)
    x_1 = BatchNormalization(axis = bn_axis, name = base_name + 'bn_1')(x_1)
    x_1 = Activation('relu')(x_1)
    
    x_2 = Conv2D(filters = filters, 
                 kernel_size = k_2, 
                 strides = (1, 1), 
                 padding = 'same', 
                 name = base_name + 'conv_2')(inputs)
    x_2 = BatchNormalization(axis = bn_axis, name = base_name + 'bn_2')(x_2)
    x_2 = Activation('relu')(x_2)
    
    x_c = Concatenate(axis = bn_axis)([x_1, x_2])
#    The shape of x_c: b, h, w, filters * 2
    
    x = squeeze_excite_block(x_c, ratio, block)
#    The shape of x: b, h, w, filters * 2
    x = Conv2D(filters = filters, 
               kernel_size = 1, 
               strides = (1, 1), 
               name = base_name + 'conv_3')(x)
#    The shape of x: b, h, w, filters
    x = BatchNormalization(axis = bn_axis, name = base_name + 'bn_3')(x)
    x = Activation('relu')(x)
    
    return x

def _deep_supervision(inputs, filters, block):
    base_name = 'ds_block_' + str(block) + '_'
    x = Conv2D(filters = filters, 
               kernel_size = (1, 1), 
               strides = (1, 1), 
               name = base_name + 'conv_1')(inputs)
    x = BatchNormalization(axis = bn_axis, name = base_name + 'bn_1')(x)
    x = Activation('relu')(x)
#    The shape of x: b, h, w, filters
    
    x = GlobalAveragePooling2D()(x)
#    The shape of x: b, 1, filters
    x = Dense(filters, activation = 'relu', name = base_name + 'dense_1')(x)
    
    return x

def _initial_conv_block(inputs):
    x = Conv2D(filters = 32, 
               kernel_size = (7, 7), 
               strides = (2, 2), 
               padding = 'same', 
               name = 'init_conv')(inputs)
    x = BatchNormalization(axis = bn_axis, name = 'init_conv_bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size = (3, 3), 
                     strides = (2, 2), 
                     padding = 'same', 
                     name = 'init_MaxPool')(x)
    
    return x

def _transition_layer(inputs, filters, strides, block):
    base_name = 'tl_' + str(block) + '_'
    x = Conv2D(filters = filters, 
               kernel_size = (1, 1), 
               strides = strides, 
               padding = 'same', 
               name = base_name + 'conv_1')(inputs)
    x = BatchNormalization(axis = bn_axis, name = base_name + 'bn_1')(x)
    x = Activation('relu')(x)
    
    return x

def _SK_Net_Plus(input_shape, classes):
    inputs = Input(shape = input_shape)
#    The shape of inputs: 224 x 224 x 3
    
    x_1 = _initial_conv_block(inputs = inputs)
#    The shape of x_1: 56 x 56 x 32
    
    t_1 = _transition_layer(x_1, 64, 1, 1)
#    The shape of t_1: 56 x 56 x 64
    
    x_2 = _select_kernel(t_1, [3, 5], 64, 4, 1)
#    The shape of x_2: 56 x 56 x 64
    
    a_1 = add([t_1, x_2])
#    The shape of a_1: 56 x 56 x 64
    
    p_1 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(a_1)
#    The shape of p_1: 28 x 28 x 64
    
    t_2 = _transition_layer(p_1, 128, 1, 2)
#    The shape of t_2: 28 x 28 x 128
    
    x_3 = _select_kernel(t_2, [3, 5], 128, 4, 2)
#    The shape of x_3: 28 x 28 x 128
    
    a_2 = add([t_2, x_3])
#    The shape of a_2: 28 x 28 x 128
    
    p_2 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(a_2)
#    The shape of p_2: 14 x 14 x 128
    
    t_3 = _transition_layer(p_2, 256, 1, 3)
#    The shape of t_3: 14 x 14 x 256
    
    x_4 = _select_kernel(t_3, [3, 5], 256, 4, 3)
#    The shape of x_4: 14 x 14 x 256
    
    a_3 = add([t_3, x_4])
#    The shape of a_3: 14 x 14 x 256
    
    p_3 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(a_3)
#    The shape of p_3: 7 x 7 x 256
    
    t_4 = _transition_layer(p_3, 512, 1, 4)
#    The shape of t_4: 7 x 7 x 512
    
    x_5 = _select_kernel(t_4, [1, 3], 512, 4, 4)
#    The shape of x_5: 7 x 7 x 512
    
    a_4 = add([t_4, x_5])
#    The shape of a_4: 7 x 7 x 512
    
    x_6 = _select_kernel(a_4, [1, 3], 512, 4, 5)
#    The shape of x_6: 7 x 7 x 512
    
    a_5 = add([a_4, x_6])
#    The shape of a_5: 7 x 7 x 512
    
    ds_1 = _deep_supervision(a_2, 128, 1)
    ds_2 = _deep_supervision(a_3, 128, 2)
    ds_3 = _deep_supervision(a_4, 128, 3)
    ds_4 = _deep_supervision(a_5, 128, 4)
#    The shape of ds_1, ds_2, ds_3, ds_4: 1 x 128
    
    output = Concatenate(axis = bn_axis)([ds_1, ds_2, ds_3, ds_4])
#    The shape of output: 1 x 512
    output = Dropout(0.5)(output)
    output = Dense(classes, activation = 'softmax', name = 'fc_2')(output)
    
    model = Model(inputs = inputs, outputs = output, name = 'SK_Net')
    
    return model 
    
if __name__ == '__main__':
    model = _SK_Net_Plus((224, 224, 3), 10)
    plot_model(model, to_file = 'model_SK.png', show_shapes = True, show_layer_names = True)
    print(len(model.layers))
    model.summary()     
    

