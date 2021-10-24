import numpy as np


"""
Forward pass for a dense layer:

Takes the inputs, weights, and biases
Returns the dot product of inputs and weights and adds biases

What is the dot product?
dot_product = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
Both vectors need to be the same size

Why do we use it?
We multiply values in a matrix of inputs by the corresponding value in a matrix of weights
"""


def dense_forward(inputs, weights, biases):
    return np.dot(inputs, weights) + biases


"""
Forward pass through a rectified linear (relu) activation function

This returns y=x when x > than 1, otherwise 0 

We apply this activation function to dense layers
"""


def relu_forward(inputs):
    return np.maximum(0, inputs)