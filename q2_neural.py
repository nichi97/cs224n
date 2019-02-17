#!/usr/bin/env python

import numpy as np
import random

from importlib import reload

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
import q2_gradcheck
reload(q2_gradcheck)

gradcheck_naive = q2_gradcheck.gradcheck_naive


def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)

    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.
    ## YOUR CODE HERE: forward propagation
    z1 = np.matmul(X, W1) + b1
    h = sigmoid(z1)
    z2 = np.matmul(h, W2) + b2
    y_hat = softmax(z2)
    y = labels
    # what is wrong with my cost function?
    cost = -np.sum(y * np.log(y_hat)) / X.shape[0]
    #cost = np.sum(-np.log(y_hat[labels==1])) / X.shape[0]
    #### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    #delta1 = y_hat - labels
    #s2 = sigmoid(z2)
    # delta2 = delta1 * sigmoid_grad(s2)
    # delta3 = np.matmul(delta2, W2.T)
    # s1 = sigmoid(z1)
    # delta4 = delta3 * sigmoid_grad(s1)
    # gradW1 = np.matmul(X.T, delta4)
    # gradb1 = np.sum(delta4, axis = 0)
    # gradW2 = np.matmul(h2.T, delta2)
    # gradb2 = np.sum(delta2, axis = 0)

    # why do we need to divide by X.shape?
    delta1 = (y_hat - labels) / X.shape[0]
    delta2 = np.matmul(delta1, W2.T)
    delta3 = delta2 * sigmoid_grad(h)
    gradW2 = np.matmul(h.T, delta1)
    gradb2 = np.sum(delta1, axis = 0)
    gradW1 = np.matmul(X.T, delta3)
    gradb1 = np.sum(delta3, axis = 0)


    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")
    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)

    return


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    #your_sanity_checks()
