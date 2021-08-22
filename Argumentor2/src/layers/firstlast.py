'''
Created on Feb 27, 2017

@author: judeaax
'''
from keras.engine.topology import Layer
import keras.backend as K
from keras.backend import tensorflow_backend as tf
from typing import List
from keras.layers.core import Permute
import numpy


class LastTime(Layer):
    
    def __init__(self, max_len, **kwargs):
        self.max_len = max_len
        super(LastTime, self).__init__(**kwargs)
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])
    
    def call(self, x, mask=None):
        
        return x[:, self.max_len - 1, :]


class FirstTime(Layer):
    
    def __init__(self, max_len, **kwargs):
        self.max_len = max_len
        super(FirstTime, self).__init__(**kwargs)
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])
    
    def call(self, x, mask=None):
        return x[:, 0, :]
    
class TimeSum(Layer):
    
    def __init__(self, **kwargs):
        super(TimeSum, self).__init__(**kwargs)
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])
    
    def call(self, x, mask=None):
        return K.sum(x, 1) / 13
