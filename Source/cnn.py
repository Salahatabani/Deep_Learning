from builtins import object
import numpy as np

from Source.layers import *
from Source.Fast_CNNs.fast_layers import *
from Source.layer_utils import *
from Source.Fast_CNNs.fast_layers import conv_forward_fast, conv_backward_fast
from Source.Fast_CNNs.fast_layers import max_pool_forward_fast, max_pool_backward_fast


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        self.params['W1'] = weight_scale*np.random.randn(num_filters,input_dim[0],filter_size,filter_size)
        self.params['W2'] = weight_scale*np.random.randn(int(num_filters*input_dim[1]*input_dim[2]/4),hidden_dim)
        self.params['W3'] = weight_scale*np.random.randn(hidden_dim,num_classes)
        self.params['b1'] =  np.zeros((num_filters))
        self.params['b2'] =  np.zeros((hidden_dim))
        self.params['b3'] =  np.zeros((num_classes))

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        
        # Slow forward pass using the naive functions that I developed
        """h1,cache1 = conv_forward_naive(X, self.params['W1'], self.params['b1'], conv_param)"""  
        # fast conv forward pass using the Cython extension 
        h1,cache1 = conv_forward_fast(X, self.params['W1'], self.params['b1'], conv_param)
        relu1, cache_relu1 = relu_forward(h1) 
        #Slow maxpool forward pass using the naive functions that I developed
        """mxpl, cache_mxpl =  max_pool_forward_naive(relu1, pool_param)""" 
        # fast maxpool forward pass using the Cython extension
        mxpl, cache_mxpl =  max_pool_forward_fast(relu1, pool_param) 
        aff_relu, cache_ar = affine_relu_forward(mxpl, self.params['W2'], self.params['b2'])
        scores, cache3 = affine_forward(aff_relu, self.params['W3'], self.params['b3'])
             

        if y is None:
            return scores

        loss, grads = 0, {}

        loss,d_softmax = softmax_loss(scores, y)
        loss += self.reg*0.5*(np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2) + np.sum(self.params['W3']**2))
        
        # Slow backward propagation using the naive functions that I developed
        
        dout, grads['W3'], grads['b3'] = affine_backward(d_softmax, cache3)
        dout, grads['W2'], grads['b2'] = affine_relu_backward(dout, cache_ar)
        
        # Slow maxpool backward propagation using the naive functions that I developed
        """"dout = max_pool_backward_naive(dout, cache_mxpl)"""
        # fast maxpool backward propagation using the Cython extension 
        dout =  max_pool_backward_fast(dout, cache_mxpl)
        dout = relu_backward(dout, cache_relu1)
        # Slow conv backward propagation using the naive functions that I developed
        """dout, grads['W1'], grads['b1'] = conv_backward_naive(dout, cache1)"""
        # fast conv backward propagation using the Cython extension 
        dout, grads['W1'], grads['b1'] = conv_backward_fast(dout, cache1)
        
        #####
        grads['W1'] += self.reg*np.sum(self.params['W1'])
        grads['W2'] += self.reg*np.sum(self.params['W2'])
        grads['W3'] += self.reg*np.sum(self.params['W3'])


        return loss, grads
