import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  der = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
          
      '''
      To calc gradient, there are three cases
      1) Margin is 0 -> gradient is always 0
      2) Gradient with respect to correct label (just dW += -X[i])
      3) Gradient with respect to incorrect labels (just dW += X[i])

      Every training example, you calculate these three things and update dW
      '''
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if j == y[i]:
        continue
      if margin > 0:
        loss += margin
        dW[:,y[i]] = dW[:,y[i]] + (-1*X[i].reshape(1,-1)) #calculate 2)
        dW[:,j] = dW[:,j] + (1*X[i].reshape(1,-1)) #calculte 1)
      else:
          #calculate 1)
          dW[:,j] += 0 #isnt need, but reminds me whats doing on
          dW[:,y[i]] += 0


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg*W #derivate of W*W = W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """

  #a vectorized version of the structured SVM loss
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]

  #get the scores for every datapoint                   
  scores = X.dot(W)

  #using "advanced slicing" to get a 1D vector of the correct scores PER EXAMPLE. So select from scores the index that corresponds to y matrix.
  # this produces a num_train, 1 size matrix where each item in vector corresponds to the correct score for that training example
  correct_scores = scores[np.arange(scores.shape[0]), y]

  #calculate the margins by subtracting all scores from correct scores + delta
  margin = scores - correct_scores.reshape(num_train,1) + 1

  #get rid of instances where margin is negative
  b = margin <= 0
  margin[b] = 0

  #get rid of instances where margin is 1 (you dont sum correct label)
  c = margin == 1.0
  margin[c] = 0

  #sum loss and normalize
  loss = np.sum(margin)
  loss /= num_train
  
  #regularization 
  loss += reg * np.sum(W * W)


  #a vectorized version of the gradient for the structured SVM loss

  newMatrix = margin
  
  #we need to know where spot where the margins are greater than 0. This will produce a matrix where each position in the array
  # corresponds to the margin being greater than 0 or not. 
  newMatrix[newMatrix > 0] = 1

  #You need to sum how many times the margin is greater than 0. As a 1D vector the size of the classes.
  counts = np.sum(newMatrix,axis=1)

  #using the "advanced slicing" method, you need to update each position in the newMatrix to be equal to the negative of counts where
  # This is only assigned to the correct labels for each image. You do not need to set anything to all the other places in the matrix because 
  # the equation for the derivative. the correct label requires the sum (from counts), where the incorrect label does not require a sum. 
  newMatrix[np.arange(scores.shape[0]), y] = -(counts.reshape(1,-1))
  
  #now that the coefficinets for newMatrix are correct, multiplying the newMatrix by the transposed data will result in the correct dW equation for both 
  # correct labels and incorrect labels
  dW = X.T.dot(newMatrix)

  #normalize dW again
  dW /= num_train
  dW += reg*W #derivate of W*W = W


  return loss, dW
