'''
Created on Feb 24, 2017

@author: judeaax
'''
from keras.engine.topology import Layer
import keras
import keras.backend as K

class Reversed(Layer):
    """
    Reverses its input along the specified axis.
    
    Example: 
        layer = Reversed(1)(input)
    """
    def __init__(self, axis, **kwargs):
        self.supports_masking = True
        self.axis=axis
        super(Reversed, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return K.reverse(x, self.axis)
    