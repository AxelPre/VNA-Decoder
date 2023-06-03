import numpy as np
from random import random
from numba import njit
import random as rand
import tensorflow as tf

class RNN_test():
    def __init__(self, n_input, n_output, n_hidden):
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden




class MDTensorizedRNNCell(tf.compat.v1.nn.rnn_cell.RNNCell):
    """The 2D Tensorized RNN cell.
    """
    def __init__(self, num_units = None, activation = None, name=None, dtype = None, reuse=None):
        super(MDTensorizedRNNCell, self).__init__(_reuse=reuse, name=name)
        # save class variables
        self._num_in = 2
        self._num_units = num_units
        self._state_size = num_units
        self._output_size = num_units
        self.activation = activation

        # set up input -> hidden connection
        self.W = tf.get_variable("W_"+name, shape=[num_units, 2*num_units, 2*self._num_in],
                                    initializer=slim.xavier_initializer(), dtype = dtype)

        self.b = tf.get_variable("b_"+name, shape=[num_units],
                                    initializer=slim.xavier_initializer(), dtype = dtype)

    @property
    def input_size(self):
        return self._num_in # real

    @property
    def state_size(self):
        return self._state_size # real

    @property
    def output_size(self):
        return self._output_size # real

    def call(self, inputs, states):

        inputstate_mul = tf.einsum('ij,ik->ijk', tf.concat(states, 1),tf.concat(inputs,1))
        # prepare input linear combination
        state_mul = tensordot(tf, inputstate_mul, self.W, axes=[[1,2],[1,2]]) # [batch_sz, num_units]

        preact = state_mul + self.b

        output = self.activation(preact) # [batch_sz, num_units] C

        new_state = output

        return output, new_state

