'''
Created on Aug 16, 2016

@author: Alex Judea
'''
import numpy
from requests.sessions import session

from keras import initializations, constraints, regularizers
from keras.engine.topology import Layer
from keras.layers.embeddings import Embedding

import keras.backend as K

class CorrectedEmbedding(Embedding):

    def get_output_shape_for(self, input_shape):
        if len(input_shape) == 1:
            return (input_shape[0], self.output_dim)
        elif len(input_shape) > 2: 
            return input_shape + (self.output_dim,)
        else:
            if not self.input_length:
                input_length = input_shape[1]
            else:
                input_length = self.input_length
            return (input_shape[0], input_length, self.output_dim)

class HybridEmbedding(Layer):
    '''
    Turn positive integers (indexes) into dense vectors of fixed size.
    eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

    This layer can only be used as the first layer in a model.
    
    In contrast to Embedding, this layer takes a matrix of learnable weights, and a matrix of fixed weights.
    '''
    
    
    def __init__(self, input_dim, output_dim, weights: [], mask, mask_zero, input_length=None, ** kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.mask_zero = mask_zero
        self.init = initializations.get('uniform')
        self.mask = mask
        kwargs['input_shape'] = (self.input_length,)
        kwargs['input_dtype'] = 'int32'
        super(HybridEmbedding, self).__init__(**kwargs)
        
        self.te = [weights[0]]
        self.ue = [weights[1]]       
        
    def set_weights(self, weights, params):
        '''Sets the weights of the layer, from Numpy arrays.

        # Arguments
            weights: a list of Numpy arrays. The number
                of arrays and their shape must match
                number of the dimensions of the weights
                of the layer (i.e. it should match the
                output of `get_weights`).
        '''
        
        if len(params) != len(weights):
            raise Exception('You called `set_weights(weights)` on layer "' + self.name + 
                            '" with a  weight list of length ' + str(len(weights)) + 
                            ', but the layer was expecting ' + str(len(params)) + 
                            ' weights. Provided weights: ' + str(weights))
        if not params:
            return
        weight_value_tuples = []
        param_values = K.batch_get_value(params)
        for pv, p, w in zip(param_values, params, weights):
            if pv.shape != w.shape:
                raise Exception('Layer weight shape ' + 
                                str(pv.shape) + 
                                ' not compatible with '
                                'provided weight shape ' + str(w.shape))
            weight_value_tuples.append((p, w))
        
        K.batch_set_value(weight_value_tuples)        

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)

    def build(self, input_shape):
        
#         self.Wt = K.variable(self.te)
#         self.Wu = K.variable(self.ue)
        self.Wt = self.init((self.input_dim, self.output_dim),
                           name='{}_Wt'.format(self.name))
        self.Wu = self.init((self.input_dim, self.output_dim),
                           name='{}_Wu'.format(self.name))
        self.M = self.init((self.input_dim, self.output_dim),
                           name='{}_M'.format(self.name))
        
        
        self.trainable_weights = [self.Wt]
        self.non_trainable_weights = [self.Wu]
        self.set_weights(self.te, self.trainable_weights)
        self.set_weights(self.ue, self.non_trainable_weights)
        self.set_weights([self.mask], [self.M])
        
        # TODO this has to be done properly
#         self.trainable_weights[0].eval(K.get_session())
#         self.non_trainable_weights[0].eval(K.get_session())
        K.eval(self.trainable_weights[0])
#         self.M.eval(K.get_session())
        K.eval(self.M)

    def call(self, x, mask=None):
        
        _mask = K.gather(self.M, x)
        gwt = K.gather(self.Wt, x)
        gwu = K.gather(self.Wu, x)
    
        return gwt * _mask + gwu

    def get_output_shape_for(self, input_shape):
        if not self.input_length:
            input_length = input_shape[1]
        else:
            input_length = self.input_length
        shape = (input_shape[0], input_length, self.output_dim)
        return shape
    

    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'input_length': self.input_length,
                  'mask_zero': self.mask_zero,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'dropout': self.dropout}
        base_config = super(Embedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
