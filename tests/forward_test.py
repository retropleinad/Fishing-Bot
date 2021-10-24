import numpy as np
from neural_network import forward


# Problem, when we do the dot product in batches, it gives us the not aligned error
# This is because we're not doing a normal dot product, and we're doing matrix multiplication,
# because we want to know the combination for each weight with each bias
def dot_test():
    a = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    b = np.array([
        [4, 5, 6],
        [7, 8, 9]
    ])
    return np.dot(a, b.T)


def basic_test():
    # A batch of inputs
    sample_inputs = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 2.0, 5.0, 1.0]
    ])

    sample_weights1 = 0.01 * np.random.randn(4, 3)
    biases1 = np.zeros((1, 3))
    sample_weights2 = 0.01 * np.random.randn(4, 3)
    biases2 = np.zeros((1, 3))

    values = forward.dense_forward(inputs=sample_inputs, weights=sample_weights1, biases=biases1)
    values = forward.relu_forward(inputs=values)
    values = forward.dense_forward(inputs=values, weights=sample_weights2, biases=biases2)
    values = forward.relu_forward(inputs=values)
    return values


print(dot_test())