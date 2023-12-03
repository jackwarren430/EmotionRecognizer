from multiprocessing.pool import ThreadPool as Pool
import numpy as np
from functools import partial

x_train = np.load("data/raw/x_train.npy")
y_train = np.load("data/raw/y_train.npy")

def initParams():
    W1r = np.random.randn(10, 13)
    W1h = np.random.randn(10, 10)
    B1 = np.random.randn(10, 1)
    W2 = np.random.randn(8, 10)
    B2 = np.random.randn(8, 1)
    return W1r, W1h, B1, W2, B2

def relu(Z):
    return np.maximum(0, Z)

def relu_prime(Z):
    return Z > 0

def softmax(Z):
    shift_z = Z - np.max(Z, 0)
    exp_scores = np.exp(shift_z)
    probabilities = exp_scores / np.sum(exp_scores, 0)
    return probabilities

def forwardProp(W1r, W1h, B1, W2, B2, X):
    #fix
    hidden_states = np.zeros((216, 10))
    hidden_layer = np.zeros((10, 1))
    for i in range(216):
        hidden_layer = relu(W1r.dot(X[i]) + W1h.dot(hidden_layer).reshape(-1) + B1.reshape(-1))
        hidden_states[i] = hidden_layer
    Z2 = W2.dot(hidden_layer) + B2
    A2 = softmax(Z2)
    return hidden_states, Z2, A2

def backwardProp(hidden_states, Z2, A2, W2, y):
    #fix
    Y = np.zeros((8,1))
    Y[int(y)][0] = 1
    dW1r, dW1h, dW2 = 0, 0, 0
    dB1, dB2 = 0, 0          
    dhidden_next = 0 
    gradient = A2 - Y
    for t in reversed(range(216)):
        

        dB2_t = 0.01
        dhidden = W2.T.dot(gradient) * relu_prime(hidden_states[t])
        dW1r_t = dhidden @ hidden_states[t].T
        dW1h_t = dhidden @ hidden_states[t-1].T if t > 0 else 0
        dB1_t = 0.011
        dW2_t = 0.011


        dW1r += dW1r_t
        dW1h += dW1h_t
        dB1 += dB1_t
        dW2 += dW2_t
        dB2 += dB2_t

    return dW1r, dW1h, dB1, dW2, dB2

def updateParams(W1r, W1h, B1, W2, B2, results, batch_size, alpha):
    W1r -= alpha * np.sum(results[:,0]) / batch_size
    W1h -= alpha * np.sum(results[:,1]) / batch_size
    B1 -= alpha * np.sum(results[:,2]) / batch_size
    W2 -= alpha * np.sum(results[:,3]) / batch_size
    B2 -= alpha * np.sum(results[:,4]) / batch_size
    return W1r, W1h, B1, W2, B2

def processSequence(sequence, weights_biases):
    x, y = sequence
    print(x.shape)
    print("----")
    W1r, W1h, B1, W2, B2 = weights_biases
    hidden_states, Z2, A2 = forwardProp(W1r, W1h, B1, W2, B2, x)
    dW1r, dW1h, dB1, dW2, dB2 = backwardProp(hidden_states, Z2, A2, W2, y)
    return [dW1r, dW1h, dB1, dW2, dB2]

def gradient_descent(X, Y, iterations, alpha):
    W1r, W1h, B1, W2, B2 = initParams()
    batch_size = 30
    for i in range(iterations):
        for j in range(int(300/batch_size)):
            x_batch = X[j*batch_size : (j*batch_size)+batch_size]
            y_batch = Y[j*batch_size : (j*batch_size)+batch_size]
            weights_biases = [W1r, W1h, B1, W2, B2]
            partial_function = partial(processSequence, weights_biases=weights_biases)
            with Pool(processes=batch_size) as pool:
                results = pool.map(partial_function, zip(x_batch, y_batch))
            W1r, W1h, B1, W2, B2 = updateParams(W1r, W1h, B1, W2, B2, results, batch_size, alpha)
    return W1r, W1h, B1, W2, B2

gradient_descent(x_train, y_train, 10, 0.1)














