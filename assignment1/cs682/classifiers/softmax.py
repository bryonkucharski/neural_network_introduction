import numpy as np
from random import shuffle

def softmax_loops(scores):
    sum = 0.0
    exp_scores = np.exp(scores)

    for i in range(len(scores)):
      sum += exp_scores[i]
    
    probs = []
    for j in range(len(scores)):
      probs.append(exp_scores[j]/sum) 

    return np.array(probs)
  
def softmax_noloops(scores):
    exp_scores = np.exp(scores)
    sum = np.sum(exp_scores,axis=1)
    return exp_scores/sum.reshape(-1,1)

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_features = X.shape[1]

  for i in range(num_train):
    scores = X[i].dot(W)
    probs = softmax_loops(scores)
    Li = -np.log(probs[y[i]])
    loss += Li

    #iterate through the classes to calculate the derivate of the loss function with respect to each weight. The derivate 
    # of the softmax is (probability - 1)*X for the correct label, and equal to probability*X if incorrect label
    dprobs = probs
    for j in range(num_classes):
      
      #subtract one if correct weight
      if j == y[i]:
        dprobs[j] = dprobs[j] - 1
      
      #multiply prob*X to get dW for this class
      dW[:, j] += (dprobs[j] * X[i])

  #normalize dW
  dW /= num_train
  dW += reg*W #derivate of W*W = W

  #normalize loss
  loss /= num_train
  loss += reg * np.sum(W * W) 

  


  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_features = X.shape[1]

  scores = X.dot(W)
  probs = softmax_noloops(scores)
  log_probs = -np.log(probs[np.arange(num_train),y])
  loss = np.sum(log_probs)

  dprobs = probs
  dprobs[np.arange(num_train),y] -= 1
  
  dW = X.T.dot(dprobs)
  
  #normalize loss
  loss /= num_train
  loss += reg * np.sum(W * W)

  #normalize dW
  dW /= num_train
  dW += reg*W 

  return loss, dW

