# My Work
## cnn.py
ThreeLayerConvNet class
Architecture of the networ:
conv - relu - 2x2 max pool - affine - relu - affine - softmax
## fc_net.py
### Class *TwoLayerNet(object)*
is a two layer implementation of fully connected network.
### Class *FullyConnectedNet(object)*
is a fully connected neural network with arbitrary number of layers
## layers.py
My implementation of: *affine_forward,affine_forward,affine_backward,relu_forward,relu_backward,batchnorm_forward,batchnorm_backward,dropout_forward,dropout_backward,conv_forward_naive,conv_backward_naive,max_pool_forward_naive,max_pool_backward_naive,spatial_batchnorm_forward,spatial_batchnorm_backward*
## optim.py
My implementation of several optimization techniques:
*agd, sgd_momentum, rmsprop, and adam*

# Imported code

## Fast_CNNs.py
Fast vectorized implementation of *conv_forward*,  *conv_backward*, *max_pool_forward*, and *max_pool_backward*.
## data_utils.py
Functions to load the CIFAR-10 data.
## layer-utils.py
Convenience layers:
*affine_relu_forward*
*affine_relu_backward*
*conv_relu_forward*
*conv_relu_pool_backward*
## solver.py
Class that uses the optimization techniques in *optim.py* to solve the optimization problem. 
