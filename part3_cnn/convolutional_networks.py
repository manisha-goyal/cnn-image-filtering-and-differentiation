"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
import numpy as np
from fully_connected_networks import Linear_ReLU, Linear, Solver, ReLU, softmax_loss

def hello_convolutional_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')


###############################################################################
###############################################################################
###                   Coding Assignment Begins Here                         ###
###                           (Look for TODO)                               ###
###############################################################################
###############################################################################

class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modfiy the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the convolutional forward pass.                  #
        # Hint: you can use function torch.nn.functional.pad for padding.  #
        # You are NOT allowed to use anything in torch.nn in other places. #
        ####################################################################
        # Replace "pass" statement with your code
        
        # Get input dimensions and parameters
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        
        # Calculate output spatial dimensions
        H_prime = 1 + (H + 2 * pad - HH) // stride
        W_prime = 1 + (W + 2 * pad - WW) // stride

        #Pad the input with zeros (left, right, top, bottom)
        x_padded = torch.nn.functional.pad(x, ([pad] * 4))

        # Initialize the output tensor
        out = torch.zeros((N, F, H_prime, W_prime), dtype=x.dtype, device=x.device)

        # Perform convolution
        for n in range(N):  # for each data point
            for f in range(F):  # for each filter
                for i in range(H_prime):  # for each output row
                    for j in range(W_prime):  # for each output column
                        # Extract the current patch from the padded input
                        h_start, w_start = i * stride, j * stride
                        x_patch = x_padded[n, :, h_start:h_start + HH, w_start:w_start + WW]

                        # Perform convolution (element-wise multiply and sum) and add bias
                        out[n, f, i, j] = torch.sum(x_patch * w[f]) + b[f]
                                        
        #####################################################################
        #                          END OF YOUR CODE                         #
        #####################################################################
        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None
        ###############################################################
        # TODO: Implement the convolutional backward pass.            #
        ###############################################################
        # Replace "pass" statement with your code
        
        # Unpack cache and get input dimensions
        x, w, b, conv_param = cache
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        _, _, H_prime, W_prime = dout.shape
        stride, pad = conv_param['stride'], conv_param['pad']

        # Initialize gradients
        dx = torch.zeros_like(x)
        dw = torch.zeros_like(w)
        db = torch.zeros_like(b)

        # Pad x and dx to handle the borders
        x_padded = torch.nn.functional.pad(x, ([pad] * 4))
        dx_padded = torch.zeros_like(x_padded)

        # Compute db (sum dout over the spatial dimensions and batch)
        db = dout.sum(dim=(0, 2, 3))

        # Compute gradients
        for n in range(N):  # for each data point
            for f in range(F):  # for each filter
                for i in range(H_prime):  # for each output row
                    for j in range(W_prime):  # for each output column
                        # Extract the current patch from the padded input
                        h_start, w_start = i * stride, j * stride
                        x_patch = x_padded[n, :, h_start:h_start + HH, w_start:w_start + WW]
                        
                        # Compute dw (accumulate the gradient with respect to w)
                        dw[f] += x_patch * dout[n, f, i, j]
                        
                        # Compute dx (propagate the gradients to the input)
                        dx_padded[n, :, h_start:h_start + HH, w_start:w_start + WW] += w[f] * dout[n, f, i, j]
                        
        # Remove padding from dx
        dx = dx_padded[:, :, pad:pad + H, pad:pad + W]

        ###############################################################
        #                       END OF YOUR CODE                      #
        ###############################################################
        return dx, dw, db


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the max-pooling forward pass                     #
        ####################################################################
        # Replace "pass" statement with your code
        
        # Get input dimensions and parameters
        N, C, H, W = x.shape
        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']
        stride = pool_param['stride']

        # Calculate output spatial dimensions
        H_prime = 1 + (H - pool_height) // stride
        W_prime = 1 + (W - pool_width) // stride

        # Initialize output tensor
        out = torch.zeros((N, C, H_prime, W_prime), dtype=x.dtype, device=x.device)

        # Perform the max-pooling operation
        for i in range(H_prime):  # for each output row
            for j in range(W_prime):  # for each output column
                # Extract the current pooling region from the input
                h_start = i * stride
                w_start = j * stride
                x_pool = x[:, :, h_start:h_start+pool_height, w_start:w_start+pool_width]
                  
                # Apply max-pooling over height and width
                out[:, :, i, j] = torch.max(torch.max(x_pool, dim=2).values, dim=2).values

        ####################################################################
        #                         END OF YOUR CODE                         #
        ####################################################################
        cache = (x, pool_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """
        dx = None
        #####################################################################
        # TODO: Implement the max-pooling backward pass                     #
        #####################################################################
        # Replace "pass" statement with your code
        
        # Unpack cache and get input dimensions
        x, pool_param = cache
        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']
        stride = pool_param['stride']
        N, C, H, W = x.shape

        # Calculate output spatial dimensions
        H_prime = 1 + (H - pool_height) // stride
        W_prime = 1 + (W - pool_width) // stride

        # Initialize dx as a tensor of zeros with the same shape as x
        dx = torch.zeros_like(x)

        # Compute gradients
        for i in range(H_prime):  # for each output row
            for j in range(W_prime):  # for each output column
                # Extract the current pooling region from the input
                h_start = i * stride
                h_end = h_start + pool_height
                w_start = j * stride
                w_end = w_start + pool_width
                x_pool = x[:, :, h_start:h_end, w_start:w_end]

                # Find the maximum value in the pooling region
                max_in_pool = torch.max(x_pool.reshape(N, C, -1), dim=2).values.reshape(N, C, 1, 1)
                
                # Create a mask to identify the maximum values
                mask = (x_pool == max_in_pool)
                
                # Reshape dout to match the shape of x_pool
                dout_reshaped = dout[:, :, i, j].reshape(N, C, 1, 1)

                # Propagate the gradient only to the maximum values
                dx[:, :, h_start:h_end, w_start:w_end] += mask * dout_reshaped
    
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################
        return dx


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ######################################################################
        # TODO: Initialize weights，biases for the three-layer convolutional #
        # network. Weights should be initialized from a Gaussian             #
        # centered at 0.0 with standard deviation equal to weight_scale;     #
        # biases should be initialized to zero. All weights and biases       #
        # should be stored in thedictionary self.params. Store weights and   #
        # biases for the convolutional layer using the keys 'W1' and 'b1';   #
        # use keys 'W2' and 'b2' for the weights and biases of the hidden    #
        # linear layer, and key 'W3' and 'b3' for the weights and biases of  #
        # the output linear layer                                            #
        #                                                                    #
        # IMPORTANT: For this assignment, you can assume that the padding    #
        # and stride of the first convolutional layer are chosen so that     #
        # **the width and height of the input are preserved**. Take a        #
        # look at the start of the loss() function to see how that happens.  #
        ######################################################################
        # Replace "pass" statement with your code
        
        # Get input dimensions
        C, H, W = input_dims
    
        # Initialize weights and biases for the convolutional layer
        self.params['W1'] = torch.normal(mean=0.0, std=weight_scale, size=(num_filters, C, filter_size, filter_size), dtype=dtype, device=device)
        self.params['b1'] = torch.zeros(num_filters, dtype=dtype, device=device)

        # After the conv-relu-pool layer, spatial dimensions will be halved due to 2x2 max pooling
        pool_out_dim = num_filters * (H // 2) * (W // 2)

        # Initialize weights and biases for the hidden layer
        self.params['W2'] = torch.normal(mean=0.0, std=weight_scale, size=(pool_out_dim, hidden_dim), dtype=dtype, device=device)
        self.params['b2'] = torch.zeros(hidden_dim, dtype=dtype, device=device)

        # Initialize weights and biases for the output layer
        self.params['W3'] = torch.normal(mean=0.0, std=weight_scale, size=(hidden_dim, num_classes), dtype=dtype, device=device)
        self.params['b3'] = torch.zeros(num_classes, dtype=dtype, device=device)
        
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet.
        """
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for three-layer convolutional     #
        # net, computing the class scores for X and storing them in the      #
        # scores variable.                                                   #
        #                                                                    #
        # Remember you can use functions defined in your implementation      #
        # above                                                              #
        ######################################################################
        # Replace "pass" statement with your code
        
        # Forward pass: Convolutional layer followed by ReLU activation and max pooling
        conv_out, conv_cache = Conv_ReLU_Pool.forward(X, W1, b1, conv_param, pool_param)
        
        # Forward pass: Linear layer followed by ReLU activation
        fc_out, fc_cache = Linear_ReLU.forward(conv_out, W2, b2)
        
        # Forward pass: Linear layer
        scores, scores_cache = Linear.forward(fc_out, W3, b3)

        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################

        if y is None:
            return scores

        loss, grads = 0.0, {}
        ####################################################################
        # TODO: Implement backward pass for three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables.  #
        # Compute data loss using softmax, and make sure that grads[k]     #
        # holds the gradients for self.params[k]. Don't forget to add      #
        # L2 regularization!                                               #
        #                                                                  #
        # NOTE: To ensure that your implementation matches ours and you    #
        # pass the automated tests, make sure that your L2 regularization  #
        # does not include a factor of 0.5                                 #
        ####################################################################
        # Replace "pass" statement with your code
        
        # Compute the loss and add L2 regularization
        loss, dscores = softmax_loss(scores, y)
        loss += self.reg * (torch.sum(W1**2) + torch.sum(W2**2) + torch.sum(W3**2))

        dfc_out, dW3, db3 = Linear.backward(dscores, scores_cache)

        # Backward pass: Linear layer followed by ReLU activation
        dconv_out, dW2, db2 = Linear_ReLU.backward(dfc_out, fc_cache)

        # Backward pass: Convolutional layer followed by ReLU activation and max pooling
        _, dW1, db1 = Conv_ReLU_Pool.backward(dconv_out, conv_cache)

        # Store gradients and add regularization to the weight gradients
        grads['W1'], grads['b1'] = dW1 + 2 * self.reg * W1, db1
        grads['W2'], grads['b2'] = dW2 + 2 * self.reg * W2, db2
        grads['W3'], grads['b3'] = dW3 + 2 * self.reg * W3, db3

        ###################################################################
        #                             END OF YOUR CODE                    #
        ###################################################################

        return loss, grads


class DeepConvNet(object):
  """
  A convolutional neural network with an arbitrary number of convolutional
  layers in VGG-Net style. All convolution layers will use kernel size 3 and 
  padding 1 to preserve the feature map size, and all pooling layers will be
  max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
  size of the feature map.

  The network will have the following architecture:
  
  {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

  Each {...} structure is a "macro layer" consisting of a convolution layer,
  an optional batch normalization layer, a ReLU nonlinearity, and an optional
  pooling layer. After L-1 such macro layers, a single fully-connected layer
  is used to predict the class scores.

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  def __init__(self, input_dims=(3, 32, 32),
               num_filters=[8, 8, 8, 8, 8],
               max_pools=[0, 1, 2, 3, 4],
               batchnorm=False,
               num_classes=10, weight_scale=1e-3, reg=0.0,
               weight_initializer=None,
               dtype=torch.float, device='cpu'):
    """
    Initialize a new network.

    Inputs:
    - input_dims: Tuple (C, H, W) giving size of input data
    - num_filters: List of length (L - 1) giving the number of convolutional
      filters to use in each macro layer.
    - max_pools: List of integers giving the indices of the macro layers that
      should have max pooling (zero-indexed).
    - batchnorm: Whether to include batch normalization in each macro layer
    - num_classes: Number of scores to produce from the final linear layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights
    - reg: Scalar giving L2 regularization strength. L2 regularization should
      only be applied to convolutional and fully-connected weight matrices;
      it should not be applied to biases or to batchnorm scale and shifts.
    - dtype: A torch data type object; all computations will be performed using
      this datatype. float is faster but less accurate, so you should use
      double for numeric gradient checking.
    - device: device to use for computation. 'cpu' or 'cuda'    
    """
    self.params = {}
    self.num_layers = len(num_filters)+1
    self.max_pools = max_pools
    self.batchnorm = batchnorm
    self.reg = reg
    self.dtype = dtype
  
    if device == 'cuda':
      device = 'cuda:0'
    
    ############################################################################
    # TODO: Initialize the parameters for the DeepConvNet. All weights,        #
    # biases, and batchnorm scale and shift parameters should be stored in the #
    # dictionary self.params.                                                  #
    #                                                                          #
    # Weights for conv and fully-connected layers should be initialized        #
    # according to weight_scale. Biases should be initialized to zero.         #
    # Batchnorm scale (gamma) and shift (beta) parameters should be initilized #
    # to ones and zeros respectively.                                          #           
    ############################################################################
    # Replace "pass" statement with your code
    
    # Get input dimensions
    C, H, W = input_dims

    # Initialize weights and biases for each convolutional layer
    for i, F in enumerate(num_filters):
        # Initialize weights for the convolutional layer
        if weight_scale == "kaiming": # Use Kaiming initialization for the convolutional layer
            self.params[f'W{i+1}'] = kaiming_initializer(Din=C, Dout=F, K=3, relu=True, device=device, dtype=dtype)
        else: # Use normal distribution if not using Kaiming
            self.params[f'W{i+1}'] = torch.normal(mean=0.0, std=weight_scale, size=(F, C, 3, 3), device=device, dtype=dtype)

        # Initialize biases for the convolutional layer
        self.params[f'b{i+1}'] = torch.zeros(F, dtype=dtype, device=device)
        
        # If batch normalization is enabled, initialize gamma and beta parameters
        if self.batchnorm:
            self.params[f'gamma{i+1}'] = torch.ones(F, dtype=dtype, device=device)
            self.params[f'beta{i+1}'] = torch.zeros(F, dtype=dtype, device=device)

        # Update input channels for the next layer
        C = F

        # Halve the spatial dimensions if max pooling is applied at this layer
        if i in self.max_pools:
            H //= 2
            W //= 2

    # Calculate the flattened size for the fully connected layer
    final_input_dim = num_filters[-1] * H * W

    # Initialize the weights and biases for the output layer
    if weight_scale == "kaiming": # Use Kaiming initialization for the fully connected layer
        self.params['W_final'] = kaiming_initializer(Din=final_input_dim, Dout=num_classes, relu=True, device=device, dtype=dtype)
    else: # Use normal distribution if not using Kaiming
        self.params['W_final'] = torch.normal(mean=0.0, std=weight_scale, size=(final_input_dim, num_classes), device=device, dtype=dtype)
    
    # Initialize biases for the output layer
    self.params['b_final'] = torch.zeros(num_classes, dtype=dtype, device=device)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.batchnorm:
      self.bn_params = [{'mode': 'train'} for _ in range(len(num_filters))]
      
    # Check that we got the right number of parameters
    if not self.batchnorm:
      params_per_macro_layer = 2  # weight and bias
    else:
      params_per_macro_layer = 4  # weight, bias, scale, shift
    num_params = params_per_macro_layer * len(num_filters) + 2
    msg = 'self.params has the wrong number of elements. Got %d; expected %d'
    msg = msg % (len(self.params), num_params)
    assert len(self.params) == num_params, msg

    # Check that all parameters have the correct device and dtype:
    for k, param in self.params.items():
      msg = 'param "%s" has device %r; should be %r' % (k, param.device, device)
      assert param.device == torch.device(device), msg
      msg = 'param "%s" has dtype %r; should be %r' % (k, param.dtype, dtype)
      assert param.dtype == dtype, msg


  def save(self, path):
    checkpoint = {
      'reg': self.reg,
      'dtype': self.dtype,
      'params': self.params,
      'num_layers': self.num_layers,
      'max_pools': self.max_pools,
      'batchnorm': self.batchnorm,
      'bn_params': self.bn_params,
    }
      
    torch.save(checkpoint, path)
    print("Saved in {}".format(path))


  def load(self, path, dtype, device):
    checkpoint = torch.load(path, map_location='cpu')
    self.params = checkpoint['params']
    self.dtype = dtype
    self.reg = checkpoint['reg']
    self.num_layers = checkpoint['num_layers']
    self.max_pools = checkpoint['max_pools']
    self.batchnorm = checkpoint['batchnorm']
    self.bn_params = checkpoint['bn_params']


    for p in self.params:
      self.params[p] = self.params[p].type(dtype).to(device)

    for i in range(len(self.bn_params)):
      for p in ["running_mean", "running_var"]:
        self.bn_params[i][p] = self.bn_params[i][p].type(dtype).to(device)

    print("load checkpoint file: {}".format(path))


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the deep convolutional network.
    Input / output: Same API as ThreeLayerConvNet.
    """
    X = X.to(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params since they
    # behave differently during training and testing.
    if self.batchnorm:
      for bn_param in self.bn_params:
        bn_param['mode'] = mode
    scores = None

    # pass conv_param to the forward pass for the convolutional layer
    # Padding and stride chosen to preserve the input spatial size
    filter_size = 3
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the DeepConvNet, computing the      #
    # class scores for X and storing them in the scores variable.              #
    #                                                                          #
    # You should use the fast versions of convolution and max pooling layers,  #
    # or the convolutional sandwich layers, to simplify your implementation.   #
    ############################################################################
    # Replace "pass" statement with your code
    
    cache = {}

    # Forward pass through all the convolutional layers
    for i in range(self.num_layers - 1):
        # For the first layer, use input X. For subsequent layers, use the output from the previous layer.
        if scores is None:
            scores = X
        
        # Get weights and biases for the current layer
        W = self.params[f'W{i+1}']
        b = self.params[f'b{i+1}']
        
        # Forward pass with batch normalization (if applicable)
        if self.batchnorm:
            # Get gamma and beta parameters for the current layer
            gamma = self.params[f'gamma{i+1}']
            beta = self.params[f'beta{i+1}']

            # Get batch normalization parameters for the current layer
            bn_param = self.bn_params[i]
            
            # Apply convolutional layer, batch normalization, ReLU activation, and max pooling (if applicable)
            if i in self.max_pools:
                scores, cache[i] = Conv_BatchNorm_ReLU_Pool.forward(scores, W, b, gamma, beta, conv_param, bn_param, pool_param)
            else:
                scores, cache[i] = Conv_BatchNorm_ReLU.forward(scores, W, b, gamma, beta, conv_param, bn_param)
        else: 
            # Apply convolutional layer, ReLU activation, and max pooling (if applicable)
            if i in self.max_pools:
                scores, cache[i] = Conv_ReLU_Pool.forward(scores, W, b, conv_param, pool_param)
            else:
                scores, cache[i] = Conv_ReLU.forward(scores, W, b, conv_param)

    # Forward pass through the fully connected layer
    W_final = self.params['W_final']
    b_final = self.params['b_final']
    
    # Apply the fully connected layer
    scores, fc_cache = Linear.forward(scores, W_final, b_final)
    cache['final'] = fc_cache

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the DeepConvNet, storing the loss  #
    # and gradients in the loss and grads variables. Compute data loss using   #
    # softmax, and make sure that grads[k] holds the gradients for             #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization does not include  #
    # a factor of 0.5                                                          #
    ############################################################################
    # Replace "pass" statement with your code

    # Compute the softmax loss and its gradient
    loss, dscores = softmax_loss(scores, y)

    # Add L2 regularization to the loss
    loss += self.reg * (torch.sum(W_final ** 2) + sum(torch.sum(self.params[f'W{i+1}'] ** 2) for i in range(self.num_layers - 1)))

    # Backprop through the fully connected layer
    dx, dW_final, db_final = Linear.backward(dscores, cache['final'])

    # Store the gradients for the fully connected layer
    grads = {}
    grads['W_final'] = dW_final + 2 * self.reg * W_final
    grads['b_final'] = db_final

    # Backprop through all the convolutional layers
    for i in reversed(range(self.num_layers - 1)):

        # Get the gradients for the current layer with batch normalization (if applicable)
        if self.batchnorm:
            if i in self.max_pools: # Apply max pooling if applicable
                dx, dW, db, dgamma, dbeta = Conv_BatchNorm_ReLU_Pool.backward(dx, cache[i])
                grads[f'gamma{i+1}'] = dgamma
                grads[f'beta{i+1}'] = dbeta
            else:
                dx, dW, db, dgamma, dbeta = Conv_BatchNorm_ReLU.backward(dx, cache[i])
                grads[f'gamma{i+1}'] = dgamma
                grads[f'beta{i+1}'] = dbeta
        else:
            if i in self.max_pools: # Apply max pooling if applicable
                dx, dW, db = Conv_ReLU_Pool.backward(dx, cache[i])
            else:
                dx, dW, db = Conv_ReLU.backward(dx, cache[i])
        
        # Store the gradients for the convolutional layer
        grads[f'W{i+1}'] = dW + 2 * self.reg * self.params[f'W{i+1}']
        grads[f'b{i+1}'] = db

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


def find_overfit_parameters():
    weight_scale = 2e-3   # Experiment with this!
    learning_rate = 1e-5  # Experiment with this!
    ###########################################################
    # TODO: Change weight_scale and learning_rate so your     #
    # model achieves 100% training accuracy within 30 epochs. #
    ###########################################################
    # Replace "pass" statement with your code
    weight_scale = 1e-1
    learning_rate = 2e-3
    ###########################################################
    #                       END OF YOUR CODE                  #
    ###########################################################
    return weight_scale, learning_rate


def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
  """
  Implement Kaiming initialization for linear and convolution layers.
  
  Inputs:
  - Din, Dout: Integers giving the number of input and output dimensions for
    this layer
  - K: If K is None, then initialize weights for a linear layer with Din input
    dimensions and Dout output dimensions. Otherwise if K is a nonnegative
    integer then initialize the weights for a convolution layer with Din input
    channels, Dout output channels, and a kernel size of KxK.
  - relu: If ReLU=True, then initialize weights with a gain of 2 to account for
    a ReLU nonlinearity (Kaiming initializaiton); otherwise initialize weights
    with a gain of 1 (Xavier initialization).
  - device, dtype: The device and datatype for the output tensor.

  Returns:
  - weight: A torch Tensor giving initialized weights for this layer. For a
    linear layer it should have shape (Din, Dout); for a convolution layer it
    should have shape (Dout, Din, K, K).
  """
  gain = 2. if relu else 1.
  weight = None
  if K is None:
    ###########################################################################
    # TODO: Implement Kaiming initialization for linear layer.                #
    # The weight scale is sqrt(gain / fan_in),                                #
    # where gain is 2 if ReLU is followed by the layer, or 1 if not,          #
    # and fan_in = num_in_channels (= Din).                                   #
    # The output should be a tensor in the designated size, dtype, and device.#
    ###########################################################################
    # Replace "pass" statement with your code
    
    fan_in = Din
    std = (gain / fan_in) ** 0.5
    weight = torch.normal(mean=0.0, std=std, size=(Din, Dout), device=device, dtype=dtype)

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  else:
    ###########################################################################
    # TODO: Implement Kaiming initialization for convolutional layer.         #
    # The weight scale is sqrt(gain / fan_in),                                #
    # where gain is 2 if ReLU is followed by the layer, or 1 if not,          #
    # and fan_in = num_in_channels (= Din) * K * K                            #
    # The output should be a tensor in the designated size, dtype, and device.#
    ###########################################################################
    # Replace "pass" statement with your code
    
    fan_in = Din * K * K  # For convolutional layers, fan_in = Din * K * K
    std = (gain / fan_in) ** 0.5
    weight = torch.normal(mean=0.0, std=std, size=(Dout, Din, K, K), device=device, dtype=dtype)
    
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  return weight


###########################################################
#             Linear and Spatial Batch Norm               #
###########################################################


class BatchNorm(object):

  @staticmethod
  def forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the PyTorch
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', torch.zeros(D, dtype=x.dtype, device=x.device))
    running_var = bn_param.get('running_var', torch.zeros(D, dtype=x.dtype, device=x.device))

    out, cache = None, None
    if mode == 'train':
      #######################################################################
      # TODO: Implement the training-time forward pass for batch norm.      #
      # Use minibatch statistics to compute the mean and variance, use      #
      # these statistics to normalize the incoming data, and scale and      #
      # shift the normalized data using gamma and beta.                     #
      #                                                                     #
      # You should store the output in the variable out. Any intermediates  #
      # that you need for the backward pass should be stored in the cache   #
      # variable.                                                           #
      #                                                                     #
      # You should also use your computed sample mean and variance together #
      # with the momentum variable to update the running mean and running   #
      # variance, storing your result in the running_mean and running_var   #
      # variables.                                                          #
      #                                                                     #
      # Note that though you should be keeping track of the running         #
      # variance, you should normalize the data based on the standard       #
      # deviation (square root of variance) instead!                        # 
      # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
      # might prove to be helpful.                                          #
      #######################################################################
      # Replace "pass" statement with your code

      # Compute mean and variance
      sample_mean = torch.mean(x, dim=0)
      sample_var = torch.var(x, dim=0, unbiased=False)

      # Compute normalized x
      xmu = x - sample_mean # centered data
      sqrtvar = torch.sqrt(sample_var + eps) # standard deviation
      ivar = 1.0 / sqrtvar # inverse of standard deviation
      x_hat = xmu * ivar # normalized data

      # Scale and shift
      out = gamma * x_hat + beta 

      # Update running stats
      running_mean = momentum * running_mean + (1 - momentum) * sample_mean
      running_var = momentum * running_var + (1 - momentum) * sample_var

      # Store values in cache
      cache = (x_hat, gamma, xmu, ivar, sqrtvar, sample_var, eps)

      #######################################################################
      #                           END OF YOUR CODE                          #
      #######################################################################
    elif mode == 'test':
      #######################################################################
      # TODO: Implement the test-time forward pass for batch normalization. #
      # Use the running mean and variance to normalize the incoming data,   #
      # then scale and shift the normalized data using gamma and beta.      #
      # Store the result in the out variable.                               #
      #######################################################################
      # Replace "pass" statement with your code
      
      # Normalize using the running mean and variance
      x_hat = (x - running_mean) / torch.sqrt(running_var + eps)

      # Scale and shift
      out = gamma * x_hat + beta

      #######################################################################
      #                           END OF YOUR CODE                          #
      #######################################################################
    else:
      raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean.detach()
    bn_param['running_var'] = running_var.detach()

    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    # Don't forget to implement train and test mode separately.               #
    ###########################################################################
    # Replace "pass" statement with your code
    
    # Unpack cache and get dimensions
    x_hat, gamma, xmu, ivar, _, _, _ = cache
    N, _ = dout.shape

    # Compute gradients for scaling and shifting
    dbeta = torch.sum(dout, dim=0)
    dgamma = torch.sum(dout * x_hat, dim=0)

    # Compute gradient w.r.t. normalized input
    dx_hat = dout * gamma

    # Compute gradient w.r.t. variance
    dvar_term1 = torch.sum(dx_hat * xmu, dim=0)
    dvar_term2 = -0.5 * ivar**3
    dvar = dvar_term1 * dvar_term2

    # Compute gradient w.r.t. mean
    dmean_term1 = torch.sum(dx_hat * -ivar, dim=0) # -ivar is the derivative of sqrtvar w.r.t. xmu
    dmean_term2 = dvar * torch.mean(-2.0 * xmu, dim=0) # -2.0 * xmu is the derivative of xmu w.r.t. sample_mean
    dmean = dmean_term1 + dmean_term2

    # Compute gradient w.r.t. input x
    dx_term1 = dx_hat * ivar # dx_hat is the derivative of x_hat w.r.t. x
    dx_term2 = dvar * (2 * xmu / N) # dvar is the derivative of sample_var w.r.t. x
    dx_term3 = dmean / N # dmean is the derivative of sample_mean w.r.t. x
    dx = dx_term1 + dx_term2 + dx_term3
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta

  @staticmethod
  def backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.
    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
    
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement;                                                       #
    ###########################################################################
    # Replace "pass" statement with your code

    # Unpack variables from cache
    xhat, gamma, xmu, inv_var, sqrtvar, var, eps = cache
    N, _ = dout.shape

    # Compute gradients for scaling and shifting
    dbeta = torch.sum(dout, dim=0)
    dgamma = torch.sum(dout * xhat, dim=0)

    # Compute gradient w.r.t. normalized input
    dxhat = dout * gamma

    # Compute gradient w.r.t. input x in a single statement
    dx = (1. / N) * inv_var * (
        N * dxhat                     # Scale the gradient
        - torch.sum(dxhat, dim=0)     # Center the gradient
        - xhat * torch.sum(dxhat * xhat, dim=0)  # Adjust for the normalization
    )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


class SpatialBatchNorm(object):

  @staticmethod
  def forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (C,) giving running mean of features
      - running_var Array of shape (C,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # Replace "pass" statement with your code
    
    N, C, H, W = x.shape

    # Reshape the input into (N * H * W, C) for vanilla batch normalization
    x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, C)

    # Apply vanilla batch normalization forward pass to the reshaped input
    bn_output, cache = BatchNorm.forward(x_reshaped, gamma, beta, bn_param)

    # Reshape the output back to original shape (N, C, H, W)
    out = bn_output.reshape(N, H, W, C).permute(0, 3, 1, 2)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.
    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # Replace "pass" statement with your code
    
    N, C, H, W = dout.shape
    
    # Reshape dout to match the (N * H * W, C) for the vanilla batch normalization
    dout_reshaped = dout.permute(0, 2, 3, 1).reshape(-1, C)
    
    # Apply vanilla batch normalization backward pass to the reshaped output
    dx_reshaped, dgamma, dbeta = BatchNorm.backward(dout_reshaped, cache)
    
    # Reshape dx back to the original shape (N, C, H, W)
    dx = dx_reshaped.reshape(N, H, W, C).permute(0, 3, 1, 2)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta



################################################################################
#               Fast Implementations and Sandwich Layers                       #
################################################################################

class FastConv(object):

  @staticmethod
  def forward(x, w, b, conv_param):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
    layer.weight = torch.nn.Parameter(w)
    layer.bias = torch.nn.Parameter(b)
    tx = x.detach()
    tx.requires_grad = True
    out = layer(tx)
    cache = (x, w, b, conv_param, tx, out, layer)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    try:
      x, _, _, _, tx, out, layer = cache
      out.backward(dout)
      dx = tx.grad.detach()
      dw = layer.weight.grad.detach()
      db = layer.bias.grad.detach()
      layer.weight.grad = layer.bias.grad = None
    except RuntimeError:
      dx, dw, db = torch.zeros_like(tx), torch.zeros_like(layer.weight), torch.zeros_like(layer.bias)
    return dx, dw, db


class FastMaxPool(object):

  @staticmethod
  def forward(x, pool_param):
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width), stride=stride)
    tx = x.detach()
    tx.requires_grad = True
    out = layer(tx)
    cache = (x, pool_param, tx, out, layer)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    try:
      x, _, tx, out, layer = cache
      out.backward(dout)
      dx = tx.grad.detach()
    except RuntimeError:
      dx = torch.zeros_like(tx)
    return dx

class Conv_ReLU(object):

  @staticmethod
  def forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.
    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    out, relu_cache = ReLU.forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = ReLU.backward(dout, relu_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db


class Conv_ReLU_Pool(object):

  @staticmethod
  def forward(x, w, b, conv_param, pool_param):
    """
    A convenience layer that performs a convolution, a ReLU, and a pool.
    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    s, relu_cache = ReLU.forward(a)
    out, pool_cache = FastMaxPool.forward(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = FastMaxPool.backward(dout, pool_cache)
    da = ReLU.backward(ds, relu_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db

class Linear_BatchNorm_ReLU(object):

  @staticmethod
  def forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that performs an linear transform, batch normalization,
    and ReLU.
    Inputs:
    - x: Array of shape (N, D1); input to the linear layer
    - w, b: Arrays of shape (D2, D2) and (D2,) giving the weight and bias for
      the linear transform.
    - gamma, beta: Arrays of shape (D2,) and (D2,) giving scale and shift
      parameters for batch normalization.
    - bn_param: Dictionary of parameters for batch normalization.
    Returns:
    - out: Output from ReLU, of shape (N, D2)
    - cache: Object to give to the backward pass.
    """
    a, fc_cache = Linear.forward(x, w, b)
    a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
    out, relu_cache = ReLU.forward(a_bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for the linear-batchnorm-relu convenience layer.
    """
    fc_cache, bn_cache, relu_cache = cache
    da_bn = ReLU.backward(dout, relu_cache)
    da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
    dx, dw, db = Linear.backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

  @staticmethod
  def forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
    out, relu_cache = ReLU.forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = ReLU.backward(dout, relu_cache)
    da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

  @staticmethod
  def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
    s, relu_cache = ReLU.forward(an)
    out, pool_cache = FastMaxPool.forward(s, pool_param)
    cache = (conv_cache, bn_cache, relu_cache, pool_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    conv_cache, bn_cache, relu_cache, pool_cache = cache
    ds = FastMaxPool.backward(dout, pool_cache)
    dan = ReLU.backward(ds, relu_cache)
    da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db, dgamma, dbeta



