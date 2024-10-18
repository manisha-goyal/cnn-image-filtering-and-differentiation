"""
Implements fully connected networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from cs2271 import Solver

def hello_fully_connected_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from fully_connected_networks.py!')


class Linear(object):

    @staticmethod
    def forward(x, w, b):
        """
        Computes the forward pass for an linear (fully-connected) layer.
        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.
        Inputs:
        - x: A tensor containing input data, of shape (N, d_1, ..., d_k)
        - w: A tensor of weights, of shape (D, M)
        - b: A tensor of biases, of shape (M,)
        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """
        out = None

        N = x.shape[0]
        # x_reshaped = x.view(N, -1)  # Flattens dimensions d_1, d_2, ..., d_k
        x_reshaped = torch.reshape(x, (N, -1))

        out = torch.mm(x_reshaped, w) + b

        cache = (x, w, b)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for an linear layer.
        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, d_1, ... d_k)
          - w: Weights, of shape (D, M)
          - b: Biases, of shape (M,)
        Returns a tuple of:
        - dx: Gradient with respect to x, of shape
          (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache

        dx, dw, db = None, None, None
      
        # Compute gradients
        dx = torch.matmul(dout, w.t()).view(*x.shape)
        # dw = torch.matmul(x.view(x.shape[0], -1).t(), dout)
        dw = torch.matmul(x.reshape(x.shape[0], -1).t(), dout)
        db = torch.sum(dout, dim=0)
     
        return dx, dw, db


class ReLU(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - x: Input; a tensor of any shape
        Returns a tuple of:
        - out: Output, a tensor of the same shape as x
        - cache: x
        """
        out = None
  
        # Replace "pass" statement with your code
        out = torch.maximum(torch.tensor(0, dtype=x.dtype, device=x.device), x)
       
        cache = x
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout
        Returns:
        - dx: Gradient with respect to x
        """
        dx, x = None, cache

        # Derivative is 1 for positive elements, 0 otherwise
        dx = dout * (cache > 0).float()  
      
        return dx


class Linear_ReLU(object):

    @staticmethod
    def forward(x, w, b):
        """
        Convenience layer that performs an linear transform
        followed by a ReLU.

        Inputs:
        - x: Input to the linear layer
        - w, b: Weights for the linear layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, fc_cache = Linear.forward(x, w, b)
        out, relu_cache = ReLU.forward(a)
        cache = (fc_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-relu convenience layer
        """
        fc_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db


###############################################################################
#                                  Optimizers                                 #
###############################################################################

def sgd_momentum(w, dw, config=None):
  """
  Performs stochastic gradient descent with momentum.
  config format:
  - learning_rate: Scalar learning rate.
  - momentum: Scalar between 0 and 1 giving the momentum value.
    Setting momentum = 0 reduces to sgd.
  - velocity: A numpy array of the same shape as w and dw used to store a
    moving average of the gradients.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-2)
  config.setdefault('momentum', 0.9)
  v = config.get('velocity', torch.zeros_like(w))

  next_w = None

  v = config['momentum']*v - config['learning_rate'] * dw
  next_w = w + v
  config['velocity'] = v

  return next_w, config


def adam(w, dw, config=None):
  """
  Uses the Adam update rule, which incorporates moving averages of both the
  gradient and its square and a bias correction term.
  config format:
  - learning_rate: Scalar learning rate.
  - beta1: Decay rate for moving average of first moment of gradient.
  - beta2: Decay rate for moving average of second moment of gradient.
  - epsilon: Small scalar used for smoothing to avoid dividing by zero.
  - m: Moving average of gradient.
  - v: Moving average of squared gradient.
  - t: Iteration number.
  """
  if config is None: config = {}
  config.setdefault('learning_rate', 1e-3)
  config.setdefault('beta1', 0.9)
  config.setdefault('beta2', 0.999)
  config.setdefault('epsilon', 1e-8)
  config.setdefault('m', torch.zeros_like(w))
  config.setdefault('v', torch.zeros_like(w))
  config.setdefault('t', 0)

  next_w = None
 
  config['t'] += 1
  config['m'] = config['beta1']*config['m'] + (1-config['beta1'])*dw
  mt = config['m'] / (1-config['beta1']**config['t'])
  config['v'] = config['beta2']*config['v'] + (1-config['beta2'])*(dw*dw)
  vc = config['v'] / (1-(config['beta2']**config['t']))
  w = w - (config['learning_rate'] * mt)/ (torch.sqrt(vc) + config['epsilon'])
  next_w = w
  
  return next_w, config



###############################################################################
#                              Loss Functions                                 #
###############################################################################

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for
      the jth class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label
      for x[i] and 0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - x.max(dim=1, keepdim=True).values
    Z = shifted_logits.exp().sum(dim=1, keepdim=True)
    log_probs = shifted_logits - Z.log()
    probs = log_probs.exp()
    N = x.shape[0]
    loss = (-1.0 / N) * log_probs[torch.arange(N), y].sum()
    dx = probs.clone()
    dx[torch.arange(N), y] -= 1
    dx /= N
    return loss, dx

