from builtins import range
from builtins import object
import numpy as np

from Source.layers import *
from Source.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg


        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None

        W1 = self.params['W1']  
        b1 = self.params['b1']  
        W2 = self.params['W2']  
        b2 = self.params['b2'] 
        
        N = X.shape[0]
        shape = X.shape
        X_t = X.reshape(N,-1)
        affine1 = np.dot(X_t,W1)+b1
        relu1 = np.maximum(affine1,0)
        scores = np.dot(relu1,W2)+b2
        


        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}


        cache = (X,W2)
        loss,d_sm = softmax_loss(scores, y)
        loss = loss + 0.5*self.reg*np.sum(W1*W1) + 0.5*self.reg*np.sum(W2*W2)
        
        dx2,dw2,db2 = affine_backward(d_sm, (relu1,W2,b2))
        dx_relu = relu_backward(dx2, affine1)
        #dw_relu = relu_backward(dw2, affine1)
        #db_relu = relu_backward(db2, affine1)
        dx1,dw1,db1 = affine_backward(dx_relu, (X,W1,b1))
        grads['W1']= dw1 + self.reg*W1
        grads['b1']= db1
        grads['W2']= dw2 + self.reg*W2
        grads['b2']= db2
        

        return loss, grads


def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    b, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(b)
    cache = (fc_cache,bn_cache, relu_cache)
    return out, cache
    
def affine_bn_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    db, dgamma, dbeta = batchnorm_backward(da, bn_cache)
    dx, dw, db = affine_backward(db, fc_cache)
    return dx, dw, db, dgamma, dbeta

class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        
        
        dims = [input_dim]+hidden_dims+[num_classes]
        for i in range(self.num_layers):
            self.params['W'+str(i+1)]= weight_scale * np.random.randn(dims[i],dims[i+1])
            self.params['b'+str(i+1)] = np.zeros(dims[i+1])
        
        if use_batchnorm==True:
            self.gb = {}
            self.params['gamma'+str(len(hidden_dims))] = np.ones(hidden_dims[-1])
            self.params['beta'+str(len(hidden_dims))] = np.zeros(hidden_dims[-1])
            for i in range(1,len(hidden_dims)):
                self.params['gamma'+str(i)] = np.ones(hidden_dims[i-1])
                self.params['beta'+str(i)] = np.zeros(hidden_dims[i-1])
        

 

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed
        
        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward 
        # of the first batch normalization layer, self.bn_params[1] to the 
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        N = X.shape[0]
        X_t = X.reshape(N,-1)
        f_pass = {}
        cache = []
        if self.use_batchnorm:
            for i in range(self.num_layers-1):
               
                X_t,c = affine_bn_relu_forward(X_t, self.params['W'+str(i+1)], 
                        self.params['b'+str(i+1)], self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)], self.bn_params[i])
                if self.use_dropout: 
                    X_t,cd = dropout_forward(X_t, self.dropout_param)
                    c = c + (cd,)
                cache.append(c)
        else:
            for i in range(self.num_layers-1):
                X_t,c = affine_relu_forward(X_t, self.params['W'+str(i+1)], self.params['b'+str(i+1)])
                if self.use_dropout: 
                    X_t,cd = dropout_forward(X_t, self.dropout_param)
                    c = c + (cd,)
                cache.append(c)
        
        scores,c = affine_forward(X_t,self.params['W'+str(self.num_layers)],self.params['b'+str(self.num_layers)])
        cache.append(c)
        


        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}

        loss,d_softmax = softmax_loss(scores, y)
        for i in range(self.num_layers):
            loss = loss + 0.5*self.reg*np.sum(self.params['W'+str(i+1)]**2)
        
        dx,dw,db = affine_backward(d_softmax, cache[-1])
        dw += self.reg*self.params['W%i'%(self.num_layers)]
        grads['W'+str(self.num_layers)]= dw 
        grads['b'+str(self.num_layers)]= db
        
        if self.use_batchnorm:
            for i in range(self.num_layers-1,0,-1):
                if self.use_dropout:
                    dx = dropout_backward(dx, cache[(i-1)][-1])
                    cache[(i-1)] = cache[(i-1)][:-1]
                dx,dw,db,dgamma,dbeta = affine_bn_relu_backward(dx,cache[(i-1)])
                grads['W'+str(i)]= dw + self.reg * self.params['W%d' % i]
                grads['b'+str(i)]= db
                grads['gamma'+str(i)]= dgamma
                grads['beta'+str(i)]= dbeta
            
        else:
            for i in range(self.num_layers-1,0,-1):
                if self.use_dropout:
                    dx = dropout_backward(dx, cache[(i-1)][-1])  
                    cache[(i-1)] = cache[(i-1)][:-1]
                   
                dx,dw,db = affine_relu_backward(dx, cache[(i-1)])
                grads['W'+str(i)]= dw + self.reg * self.params['W%d' % i]
                grads['b'+str(i)]= db
                


        return loss, grads
