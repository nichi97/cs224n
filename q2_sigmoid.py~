#!/usr/bin/env python

import numpy as np


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Arguments:
    x -- A scalar or numpy array.

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))

    return s


def sigmoid_grad(s):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input s should be the sigmoid
    function value of your original input x.

    Arguments:
    s -- A scalar or numpy array.

    Return:
    ds -- Your computed gradient.
    """
    ds = s * (1 - s)

    return ds


def test_sigmoid_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print(f)
    f_ans = np.array([
        [0.73105858, 0.88079708],
        [0.26894142, 0.11920292]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print(g)
    g_ans = np.array([
        [0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)
    print("You should verify these results by hand!\n")


def test_sigmoid():
    """
    Use this space to test your sigmoid implementation by running:
        python q2_sigmoid.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print("Running your tests...")
    x1 = np.array([1,2,-1,0,0.5])
    f1 = sigmoid(x1)
    f1_ans = np.array([
        0.7310585786300049, 0.88079707797788231,
        0.2689414213699951, 0.5, 0.62245933120185459
        ])
    print(f1)
    print(f1_ans)
    assert np.allclose(f1, f1_ans, rtol = 1e-05, atol=1e-06)
    g1 = sigmoid_grad(f1)
    g1_ans = np.array([ 0.19661193, 0.10499359, 0.19661193, 
        0.25 , 0.23500371])
    print(g1)
    print(g1_ans)
    assert(np.allclose(g1, g1_ans, rtol = 1e-05, atol = 1e-06))



if __name__ == "__main__":
    test_sigmoid_basic()
    test_sigmoid()
