from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
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

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #pass
        C, H, W = input_dim
        self.params["W1"] = np.random.normal(loc=0.0,scale=weight_scale,size=(num_filters,C,filter_size,filter_size))
        self.params["W2"] = np.random.normal(loc=0.0, scale=weight_scale, size=(num_filters*H*W//4,hidden_dim))
        self.params["W3"] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes))
        self.params["b1"], self.params["b2"], self.params["b3"] = np.zeros(num_filters), np.zeros(hidden_dim), np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in annp/fast_layers.py and  #
        # annp/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #pass
        out_1, cache_1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out_2, cache_2 = affine_relu_forward(out_1, W2, b2)
        scores, cache_3 = affine_forward(out_2, W3, b3)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #pass
        loss, da = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
        d_affine_out, d_affine_w, d_affine_b = affine_backward(da, cache_3)
        d_relu_out, d_relu_w, d_relu_b = affine_relu_backward(d_affine_out, cache_2)
     #   reshaped_d_relu_out = d_relu_out.reshape(out_1.shape)
        d_conv_out, d_conv_w, d_conv_b = conv_relu_pool_backward(d_relu_out, cache_1)
        
        grads["W1"], grads["W2"], grads["W3"] = d_conv_w + self.reg * W1, d_relu_w + self.reg * W2, d_affine_w + self.reg * W3
        grads["b1"], grads["b2"], grads["b3"] = d_conv_b, d_relu_b, d_affine_b

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class MyConvNet(object):
    def __init__(self,
                input_dim=(3,32,32),
                num_filters_1=32,
                num_filters_2=64,
                hidden_dim = 100,
                filter_size=7,
                num_classes=10,
                weight_scale=1e-3,
                reg=0.0,
                dtype=np.float32,
                ):
        self.reg = reg
        self.dtype = dtype
        self.params = {}
        C, H, W = input_dim
        self.params["W1"] = np.random.normal(loc=0.0,scale=weight_scale,size=(num_filters_1,C,filter_size,filter_size))
        self.params["b1"] = np.zeros(num_filters_1)
        self.params["gamma1"] = np.ones(C)
        self.params["beta1"] = np.zeros(C)
        self.params["W2"] = np.random.normal(loc=0.0,scale=weight_scale,size=(num_filters_2, num_filters_1, filter_size, filter_size))
        self.params["b2"] = np.zeros(num_filters_2)
        self.params["gamma2"] = np.ones(num_filters_1)
        self.params["beta2"] = np.zeros(num_filters_1)
        self.params["W3"] = np.random.normal(loc=0.0,scale=weight_scale,size=(num_filters_2*H*W//16, hidden_dim))
        self.params["b3"] = np.zeros(hidden_dim)
        self.params["gamma3"] = np.ones(hidden_dim)
        self.params["beta3"] = np.zeros(hidden_dim)
        self.params["W4"] = np.random.normal(loc=0.0, scale=weight_scale, size=(hidden_dim, num_classes))
        self.params["b4"] = np.zeros(num_classes)
        self.bn_params = [{"mode": "train"}, {"mode": "train"}, {"mode": "train"}, {"mode": "train"}]
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
    def loss(self, X, y=None):
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"
        for bn_param in self.bn_params:
            bn_param["mode"] = mode

        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]
        W4, b4 = self.params["W4"], self.params["b4"]
        gamma1, beta1 = self.params["gamma1"], self.params["beta1"]
        gamma2, beta2 = self.params["gamma2"], self.params["beta2"]
        gamma3, beta3 = self.params["gamma3"], self.params["beta3"]

        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}
        scores = None
        bn_x_1,cache_bn_1 = spatial_batchnorm_forward(X,gamma1,beta1,self.bn_params[0])
        out_1,cache_1 = conv_relu_pool_forward(bn_x_1,W1,b1,conv_param,pool_param)
        bn_x_2, cache_bn_2 = spatial_batchnorm_forward(out_1,gamma2,beta2,self.bn_params[1])
        out_2,cache_2 = conv_relu_pool_forward(bn_x_2,W2,b2,conv_param,pool_param)
        out_3, cache_3 = affine_bn_relu_forward(out_2, W3, b3, gamma3, beta3, self.bn_params[3])
        scores,cache_4 = affine_forward(out_3,W4,b4)

        if y is None:
            return scores
        
        loss, grads = 0, {}

        loss, da = softmax_loss(scores,y)
        loss += 0.5*self.reg*(np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3)+np.sum(W4*W4))
        d_affine_out, d_affine_w, d_affine_b = affine_backward(da,cache_4)
        d_relu_out, d_relu_w, d_relu_b, dgamma3, dbeta3 = affine_bn_relu_backward(d_affine_out,cache_3)
        reshaped_d_relu_out = d_relu_out.reshape(out_2.shape)
        d_conv_out_2, d_conv_w_2, d_conv_b_2 = conv_relu_pool_backward(reshaped_d_relu_out,cache_2)
        d_spatial_out_2, dgamma2, dbeta2 = spatial_batchnorm_backward(d_conv_out_2,cache_bn_2)
        reshaped_d_spatial_out_2 = d_spatial_out_2.reshape(out_1.shape)
        d_conv_out_1, d_conv_w_1, d_conv_b_1 = conv_relu_pool_backward(reshaped_d_spatial_out_2,cache_1)
        d_spatial_out_1, dgamma1, dbeta1 = spatial_batchnorm_backward(d_conv_out_1,cache_bn_1)

        grads["W1"] = d_conv_w_1 + self.reg*W1
        grads["W2"] = d_conv_w_2 + self.reg*W2
        grads["W3"] = d_relu_w + self.reg*W3
        grads["W4"] = d_affine_w + self.reg*W4
        grads["b1"], grads["b2"], grads["b3"], grads["b4"] = d_conv_b_1, d_conv_b_2, d_relu_b, d_affine_b
        grads["gamma1"], grads["gamma2"], grads["gamma3"] = dgamma1, dgamma2, dgamma3
        grads["beta1"], grads["beta2"], grads["beta3"] = dbeta1, dbeta2, dbeta3
        return loss, grads