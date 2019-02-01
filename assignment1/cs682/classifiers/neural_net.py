from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None,use_dropout = False,dropout_percent = 0.5,reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass

    z1 = X.dot(W1) + b1
    a1 = np.maximum(0, z1) #activation function

    if use_dropout:
      ## Dropout
      D1 = np.random.rand(a1.shape[0], a1.shape[1])                         
      D1 = D1 < dropout_percent                                        
      a1 = a1*D1                                       
      a1 = a1/dropout_percent                                     

    scores = a1.dot(W2) + b2
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    #same as my function softmax_noloops from softmax.py 
    exp = np.exp(scores)
    sum = np.sum(exp,axis=1)
    p = exp/sum.reshape(-1,1)

    #get logs of all probabilities
    log_p = -np.log(p[np.arange(N),y])

    #data loss
    loss = np.sum(log_p) / N
    #reg loss
    loss += ((reg * np.sum(W1 * W1)) + (reg *np.sum(W2 * W2)))

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################

    #compute the derivate of the softmax output
    
    dZ2 = p
    dZ2[np.arange(N),y] -= 1
    dZ2 /= N

    #give that derivate back to second layer
    dW2 = a1.T.dot(dZ2)
    db2 = np.sum(dZ2, axis=0)

    grads['W2'] = dW2
    grads['b2'] = db2

    #Calculate derivate of the output of the first layer
    dA1 = dZ2.dot(W2.T)

    if use_dropout:
      dA1 = D1 * dA1       
      dA1 = dA1/dropout_percent      

    #I now have the input to the first (hidden) layer.
    #next i need to calculate the derivate of the activation function
    #since the hidden activation is a relu, the derivate of a relu is 0 in all
    # cases where the input is less than 0
    #in this case, the input to the first (hidden) layer is a1
    dA1[a1 <= 0] = 0


    dZ1 = dA1
    dW1 = X.T.dot(dZ1)
    db1 = np.sum(dZ1, axis=0)

    grads['W1'] = dW1
    grads['b1'] = db1

    #need to backprop reg as well
    grads['W2'] += reg * W2 *2
    grads['W1'] += reg * W1 * 2
   
    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False,
            dropout_percent = 0.5,
            use_dropout = False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):



      idxs = np.random.choice(num_train, size=batch_size, replace=True)
      X_batch = X[idxs]
      y_batch = y[idxs]

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch,dropout_percent=dropout_percent, reg=reg, use_dropout=use_dropout)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W2'] -= learning_rate*grads['W2']*learning_rate_decay
      self.params['W1'] -= learning_rate*grads['W1']*learning_rate_decay
      self.params['b2'] -= learning_rate*grads['b2']*learning_rate_decay
      self.params['b1'] -= learning_rate*grads['b1']*learning_rate_decay
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    z1 = X.dot(self.params['W1']) + self.params['b1']
    a1 = np.maximum(0, z1) #activation function
    scores = a1.dot(self.params['W2']) + self.params['b2']
    y_pred = np.argmax(scores, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


