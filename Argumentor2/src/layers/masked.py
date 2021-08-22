'''
Created on Nov 21, 2016

@author: judeaax
'''
from keras.layers.core import Reshape, Lambda
import keras.backend as K
import numpy
from numpy import bool_
import theano
from keras.engine.topology import Layer
from math import nan
from keras.backend import _BACKEND

class RemoveMask(Lambda):
    """
    This layer removes a mask, if set, from the processing flow.
    
    Use it without any arguments. 
    
    Example: 
        layer = Dense(10)(RemoveMask()(previously_masked))
    """
    def __init__(self):
        super(RemoveMask, self).__init__((lambda x, mask: x))
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        return None
        

class MaskedSoftmax(Layer):
    """
    Computes a 'masked' variant of softmax. 
    
    For input value 0, the masked softmax is also zero. 
    Note that input values should only be zero, if softmax should produce zero probability for them. 
    Standard softmax produces the value 1 for input 0 (because e^0=1), usually resulting in a very low, but still non-zero probability.
    """
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskedSoftmax, self).__init__(**kwargs)
        
    def call(self, x, mask=None):
        
        nonzero = K.cast(K.not_equal(x, 0.0), "float32")
        masked_e = K.exp(x) * nonzero
        sum_masked_e = K.sum(masked_e, axis=1)
        
        scores = masked_e / (K.expand_dims(sum_masked_e, 1) + 0.01)
        
        return scores

    def get_config(self):
        return super(MaskedSoftmax, self).get_config()
    
    
