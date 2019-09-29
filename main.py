from mnist_loader import load_data_wrapper
import numpy as np
import matplotlib.pyplot as plt
import random

# Unpacks a dataset into images and labels
def vectorize_sample(sample):
    images = np.hstack([sample[k][0] for k in range(0,len(sample))])
    labels = np.hstack([sample[k][1] for k in range(0,len(sample))])
    return images, labels

# Scores the likelihood of a digit to be the digit on the image
def f(X, W1, W2, B1, B2):
    # X is a set of images
    Z1 = np.dot(W1,X) + B1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2,A1) + B2
    A2 = sigmoid(Z2)
    return A2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Determines the highest scoring digit as the digit on the image
def predict(images, W1, W2, B1, B2):
    predictions = []
    for image in images:
        a = f(image[0], W1, W2, B1, B2)
        predictions.append(np.argmax(a))
    return predictions

# Gradient Descent
def SGD(training_data, loops, sample_size, a, test_data, W1, W2, B1, B2):
    for j in range(loops):
        random.shuffle(training_data)
        for k in range(0, len(training_data), sample_size):
            sample = training_data[k:k+sample_size]
            X, Y = vectorize_sample(sample)
            # Vectorized Forward Propagation
            Z1 = np.dot(W1,X) + B1
            A1 = sigmoid(Z1)
            Z2 = np.dot(W2,A1) + B2
            A2 = sigmoid(Z2) 
            # Vectorized Back Propagation
            dZ2 = 1 / sample_size * (A2 - Y) * sigmoid_prime(Z2)
            dW2 = np.dot(dZ2, A1.T)
            dB2 = 1 / sample_size * np.sum(dZ2, axis = 1, keepdims = True)
            dZ1 = 1 / sample_size * np.dot(W2.T, dZ2) * sigmoid_prime(Z1)
            dW1 = np.dot(dZ1, X.T)
            dB1 = 1 / sample_size * np.sum(dZ1, axis = 1, keepdims = True)
            # Update Parameters
            W2 = W2 - a * dW2
            W1 = W1 - a * dW1
            B2 = B2 - a * dB2
            B1 = B1 - a * dB1
        test_results = [(np.argmax(f(x, W1, W2, B1, B2)), y) for (x, y) in test_data]
        num_correct = sum(int(x == y) for (x, y) in test_results)
        print(f'Epoch {j} : {num_correct} / {len(test_data)}')
    return W1, B1, W2, B2

# Test parameters with given data
def test(data):
    num_correct = len(data)
    for (x, y) in data:
        result = ''
        x = np.argmax(f(x, W1, W2, B1, B2))
        if x != y:
            result = 'Incorrect!'
            num_correct -= 1
        print(f'({x}, {y}) {result}')
    print(f'{num_correct} / {len(data)}')

# Load data from MNIST
training_data, validation_data, test_data = load_data_wrapper()

# Initialize parameters of 2-layer neural network with a 30-dimensional hidden layer
D = 30
W1 = np.random.randn(D,784)
W2 = np.random.randn(10,D)
B1 = np.random.randn(D,1)
B2 = np.random.randn(10,1)

# Train the parameters
W1, B1, W2, B2 = SGD(training_data, 100, 100, 2, test_data, W1, W2, B1, B2)

# Test the parameters
test(test_data[:100])