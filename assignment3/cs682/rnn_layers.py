from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    next_h = np.tanh(np.dot(x,Wx) + np.dot(prev_h,Wh) + b)
    cache = next_h, x, prev_h, Wx, Wh, b
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None

    next_h, x, prev_h, Wx, Wh, b = cache
    dtan = dnext_h * (1-(next_h**2))
    db = np.sum(dtan,axis = 0)
    dWx = np.dot(x.T,dtan)
    dx = np.dot(dtan,Wx.T)
    dWh = np.dot(prev_h.T, dtan)
    dprev_h = np.dot(dtan, Wh.T)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
   
    N, T, D = x.shape
    h_prev = h0
    H = Wh.shape[0]
    cache = {}
    h = np.zeros([N, T, H])

    #step through one unit at a time to calculate h
    for i in range(T):

      h_t,cache_t = rnn_step_forward(x[:,i,:],h_prev,Wx,Wh,b)
      h_prev = h_t

      h[:,i,:] = h_t
      cache[i] = cache_t
    
    return h, cache
 

def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H). 
    
    NOTE: 'dh' contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    N, T, H = dh.shape
    D = cache[0][1].shape[1] #hacky but yolo

    dx = np.zeros((N,T,D))
    dh0 = np.zeros((N,H))
    dWx = np.zeros((D,H))
    dWh = np.zeros((H,H))
    db = np.zeros((H))
    dnext_h = np.zeros((N,H))

    #BACKPROP THRU TIME! so you need to add all of the ders
    #start at end
    for i in reversed(range(T)):
      dnext_h += dh[:,i,:] #current derivative w.r.t h
      #take one step through commputational graph
      dx_, dnext_h, dWx_, dWh_, db_ = rnn_step_backward(dnext_h, cache[i])
      #store current T
      dx[:,i,:] = dx_

      #add to all weight ders
      dWx += dWx_
      dWh += dWh_
      db += db_

    dh0 = dnext_h #the last dh
        
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    N, T = x.shape
    D = W.shape[1]
    out = np.zeros([N, T, D])
    for i in range(N):
      sequence = x[i] #this is going to be a list of size T
      dims = W[sequence] #this will get you the embedded for each word indexed in T
      out[i] = dims #this will add the embedding to the output
    cache = x,W
    
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    x, W = cache
    V = cache[1].shape[0]
    D = dout.shape[2]
    dW = np.zeros((V,D))
    np.add.at(dW, x, dout)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """

    N, H = prev_h.shape

    #the weights for each section is stored in a single variable W.
    next_h, next_c, cache = None, None, None
    lin = np.dot(x, Wx) + np.dot(prev_h, Wh) + b

    #once transformed, the vector could be broken up into different parts
    #lecture notes say i,f,o,g
    lin_i = lin[:, 0:H] #first chunk
    lin_f = lin[:,H:2*H] #second chunk
    lin_o = lin[:,2*H:3*H] #third chunk
    lin_g = lin[:,3*H:]#last chunk

    f = sigmoid(lin_f) #decide what info to throw away

    # decide what new information we’re going to store in the cell state
    i = sigmoid(lin_i)  #Part 1, Input Gate Layer decides which values well update
    g = np.tanh(lin_g) #Part 2, tanh layer creates a vector of new candidate values that could be added to the state.

    #The previous steps already decided what to do, we just need to actually do it.
    forget = prev_c * f
    new_candidates = i * g
    next_c = forget + new_candidates

    #Finally, we need to decide what we’re going to output

    # First, we run a sigmoid layer which decides what parts of the cell state we’re going to output
    o = sigmoid(lin_o)
    #hen, we put the cell state through tanh (to push the values to be between −1 and 1) and multiply it by the output of the sigmoid gate, so that we only output the parts we decided to.
    next_h = o * np.tanh(next_c)

    cache = prev_h, prev_c, x, Wx, Wh, b, lin, i, f, o, g, next_c

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    prev_h, prev_c, x, Wx, Wh, b, lin, i, f, o, g, next_c = cache

    dnext_c += dnext_h*o*(1-np.tanh(next_c)**2)
    dprev_c = dnext_c * f

    do = (np.tanh(next_c)) * dnext_h
    df = prev_c*dnext_c
    di = dnext_c * g
    dg = dnext_c * i

    #do_unstacked = do * (sigmoid(o)*1-sigmoid(o))
    #dg_unstacked = dg * (1-np.tanh(g)**2)
    #di_unstacked = di * (sigmoid(i)*1-sigmoid(i))
    #df_unstacked = df * (sigmoid(f)*1-sigmoid(f))


    #for whatever reason you need to calc the derivates of the activations this way to get the right answers >:(
    di_unstacked = i * (1 - i) * di
    df_unstacked = f * (1 - f) * df
    do_unstacked = o * (1 - o) * do
    dg_unstacked = (1 - g **2 ) * dg


    dlin = np.hstack([di_unstacked,df_unstacked,do_unstacked,dg_unstacked])

    dx = np.dot(dlin,Wx.T)
    dWx = np.dot(x.T,dlin)

    dprev_h = np.dot(dlin,Wh.T)
    dWh = np.dot(prev_h.T,dlin)

    db = np.sum(dlin, axis = 0)






    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None

    N, T, D = x.shape
    h_prev = h0
    c_prev = 0
    H = Wh.shape[0]
    cache = {}
    h = np.zeros([N, T, H])

    # step through one unit at a time to calculate h
    for i in range(T):
        next_h, next_c, cache_t = lstm_step_forward(x[:, i, :], h_prev,c_prev, Wx, Wh, b)
        h_prev = next_h
        c_prev = next_c

        h[:, i, :] = next_h
        cache[i] = cache_t

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None


    N, T, H = dh.shape
    D = cache[0][2].shape[1]  # hacky but yolo

    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, 4*H))
    dWx = np.zeros((D, 4*H))
    dWh = np.zeros((H, 4*H))
    db = np.zeros((4*H))
    dprev_h_ = np.zeros((N, H))
    dprev_c_ = np.zeros((N, H))

    # BACKPROP THRU TIME! so you need to add all of the ders
    # start at end
    for i in reversed(range(T)):
        dprev_h_ += dh[:, i, :]  # current derivative w.r.t h
        # take one step through commputational graph
        dx_, dprev_h_, dprev_c_, dWx_, dWh_, db_ = lstm_step_backward(dprev_h_,dprev_c_, cache[i])
        # store current T
        dx[:, i, :] = dx_

        # add to all weight ders
        dWx += dWx_
        dWh += dWh_
        db += db_

    dh0 = dprev_h_  # the last dh

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
