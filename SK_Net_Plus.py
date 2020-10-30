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
from keras.layers import concatenate
from keras.layers import add
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Lambda

from keras import backend as K
from keras.models import Model

from keras.utils import plot_model

if K.image_data_format() == 'channels_first':
    bn_axis = 1
else:
    bn_axis = -1

def _grouped_conv_block(input_tensor, cardinality, output_filters, kernel_size, block):
    
    '''
    kernel_size = 3
    cardinality = 2
    '''
    
    base_name = 'ek_block_' + str(block) + '_'
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    group_list = []
    input_filters = input_tensor._keras_shape[channel_axis]
    grouped_filters = int(input_filters / cardinality)
    for c in range(cardinality):
        if K.image_data_format() == 'channels_last':
            x = Lambda(lambda z: z[:, :, :, c * grouped_filters:(c + 1) * grouped_filters])(input_tensor)
        else:
            x = Lambda(lambda z: z[:, c * grouped_filters:(c + 1) * grouped_filters, :, :])(input_tensor)
        x = Conv2D(filters = output_filters // cardinality, 
                   kernel_size = kernel_size, 
                   strides = (1, 1), 
                   padding = 'same',  
                   name = base_name + 'grouped_conv_' + str(c))(x)
        
        group_list.append(x)
        
    group_merge = concatenate(group_list, axis = channel_axis)
#    The shape of group_merge: b, h, w, output_filters
    x_c = BatchNormalization(axis = channel_axis, name = base_name + 'grouped_conv_bn')(group_merge)
    x_c = Activation('relu')(x_c)
    
    x_c = Conv2D(filters = output_filters, 
                 kernel_size = (1, 1), 
                 strides = (1, 1), 
                 name = base_name + 'mix_conv_1')(x_c)
    x_c = BatchNormalization(axis = channel_axis, name = base_name + 'mix_bn_1')(x_c)
    x_c = Activation('relu')(x_c)
    
    x_c = Conv2D(filters = output_filters, 
                 kernel_size = (1, 1), 
                 strides = (1, 1), 
                 name = base_name + 'mix_conv_2')(x_c)
    x_c = BatchNormalization(axis = channel_axis, name = base_name + 'mix_bn_2')(x_c)
    x_c = Activation('relu')(x_c)
    
    return x_c

def _select_kernel(inputs, kernels, filters, cardinality, block):
    '''
    kernels = [3, 5]
    cardinality = 2
 
    '''
    base_name = 'sk_block_' + str(block) + '_'
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    group_list = []
    input_filters = inputs._keras_shape[channel_axis]
    grouped_filters = int(input_filters / cardinality)
    for c in range(cardinality):
        if K.image_data_format() == 'channels_last':
            x = Lambda(lambda z: z[:, :, :, c * grouped_filters:(c + 1) * grouped_filters])(inputs)
        else:
            x = Lambda(lambda z: z[:, c * grouped_filters:(c + 1) * grouped_filters, :, :])(inputs)
        x_1 = Conv2D(filters = filters // cardinality, 
                     kernel_size = kernels[0], 
                     strides = (1, 1), 
                     padding = 'same', 
                     name = base_name + 'grouped_conv1_' + str(c))(x)
        group_list.append(x_1)
        x_2 = Conv2D(filters = filters // cardinality, 
                     kernel_size = kernels[1], 
                     strides = (1, 1), 
                     padding = 'same', 
                     name = base_name + 'grouped_conv2_' + str(c))(x)
        group_list.append(x_2)
    
    o_1 = add([group_list[0], group_list[2]])
    o_2 = add([group_list[1], group_list[3]])
#    The shape of o_1, o_2: b, h, w, filters // cardinality
    
    group_merge = concatenate([o_1, o_2], axis = channel_axis)
#    The shape of group_merge is: b, h, w, filters
    x_c = BatchNormalization(axis = channel_axis, name = base_name + 'grouped_conv_bn')(group_merge)
    x_c = Activation('relu')(x_c)
    
    x_c = Conv2D(filters = filters, 
                 kernel_size = (1, 1), 
                 strides = (1, 1), 
                 name = base_name + 'mix_conv_1')(x_c)
    x_c = BatchNormalization(axis = channel_axis, name = base_name + 'mix_bn_1')(x_c)
    x_c = Activation('relu')(x_c)
    
    x_c = Conv2D(filters = filters, 
                 kernel_size = (1, 1), 
                 strides = (1, 1), 
                 name = base_name + 'mix_conv_2')(x_c)
    x_c = BatchNormalization(axis = channel_axis, name = base_name + 'mix_bn_2')(x_c)
    x_c = Activation('relu')(x_c)
    
    return x_c

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

def Weakly_DenseNet(input_shape, classes):
    inputs = Input(shape = input_shape)
#    The shape of inputs: 224 x 224 x 3
    
    x_1 = _initial_conv_block(inputs)
#    The shape of x_1: 56 x 56 x 32
    x_2 = _select_kernel(x_1, [3, 5], 64, 2, 1)
#    The shape of x_2: 56 x 56 x 64
    pool_1 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x_2)
#    The shape of pool_1: 28 x 28 x 64
    x_3 = Concatenate(axis = bn_axis)([x_1, x_2])
#    The shape of x_3: 56 x 56 x 96
    x_4 = _select_kernel(x_3, [3, 5], 128, 2, 2)
#    The shape of x_4: 56 x 56 x 128
    pool_2 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x_4)
#    The shape of pool_2: 28 x 28 x 128
    x_5 = Concatenate(axis = bn_axis)([pool_1, pool_2])
#    The shape of x_5: 28 x 28 x 192
    x_6 = _select_kernel(x_5, [3, 5], 256, 2, 3)
#    The shape of x_6: 28 x 28 x 256
    pool_3 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x_6)
#    The shape of pool_3: 14 x 14 x 256
    x_7 = Concatenate(axis = bn_axis)([pool_2, x_6])
#    The shape of x_7: 28 x 28 x 384
    x_8 = _grouped_conv_block(x_7, 2, 512, 3, 4)
#    The shape of x_8: 28 x 28 x 512
    pool_4 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x_8)
#    The shape of pool_4: 14 x 14 x 512
    x_9 = Concatenate(axis = bn_axis)([pool_3, pool_4])
#    The shape of x_9: 14 x 14 x 768
    x_10 = _grouped_conv_block(x_9, 2, 512, 3, 5)
#    The shape of x_10: 14 x 14 x 512
    pool_5 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(x_10)
#    The shape of pool_5: 7 x 7 x 512
    
    output = GlobalAveragePooling2D()(pool_5)
    output = Dense(512, activation = 'relu', name = 'fc_1')(output)
    output = Dropout(rate = 0.5, name = 'dropout')(output)
    output = Dense(classes, activation = 'softmax', name = 'fc_2')(output)
    
    model = Model(inputs = inputs, outputs = output, name = 'Grouped_Weakly_Densenet_19')

    return model

if __name__ == '__main__':
    model = Weakly_DenseNet((224, 224, 3), 10)
    plot_model(model, to_file = 'model_SK_Net.png', show_shapes = True, show_layer_names = True)
    print(len(model.layers))
    model.summary()     
    

