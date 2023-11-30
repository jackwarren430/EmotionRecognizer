from multiprocessing.pool import ThreadPool as Pool
import numpy as np

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
    outputs = np.zeros((216, 8))
    hidden_states = np.zeroes((216, 10))
    hidden_layer = np.zeros((10, 1))
    for i in range(216):
        hidden_layer = relu(W1r.dot(X[i]) + W1h.dot(hidden_layer) + B1)
        hidden_states[i] = hidden_layer
    Z2 = W2.dot(hidden_layer) + B2
    A2 = softmax(Z2)
    return hidden_states, Z2, A2

def one_hot_Y(y_train):
    one_hot_Y = np.zeros((8, 300))
    for i in range(300):
        index = int(y_train[i]) - 1
        one_hot_Y[index][i] = 1
    return one_hot_Y

def backwardProp(hidden_states, Z2, A2, W2, Y):
    Y = one_hot_Y(Y)
    dW1r, dW1h, dW2 = 0, 0, 0
    dB1, dB2 = 0, 0          
    dhidden_next = 0 
    dZ2 = A2 - Y
    for i in reversed(range(216)):
        dW2_t = dZ2.dot(hidden_states[i].T)
        dB2_t = 
        dhidden = W2.T.dot(dZ2) * relu_prime(hidden_states[i])
        dW1r_t = dhidden @ sequence[t].T
        dW1h_t = dhidden @ hidden_states[t-1].T if t > 0 else 0
        dB1_t = 
        dW1r += dW1r_t
        dW1h += dW1h_t
        dB1 += dB1_t
        dW2 += dW2_t
        dB2 += dB2_t

    return dW1r, dW1h, dB1, dW2, dB2

def processSequence(X):
    global batch_dW1r, batch_dW1h, batch_dW2, batch_dB1, batch_dB2

    hidden_states, Z2, A2 = forwardProp()
    dW1r, dW1h, dB1, dW2, dB2 = backwardProp()

    batch_dW1r += dW1r
    batch_dW1h += dW1h
    batch_dB1 += dB1
    batch_dW2 += dW2
    batch_dB2 += dB2

def gradient_descent(X, Y, iterations, alpha):
    W1r, W1h, B1, W2, B2 = initParams()

    for i in range(iterations):
        batch_size = 30
        for j in range(10):
            batch_dW1r, batch_dW1h, batch_dW2 = 0, 0, 0
            batch_dB1, batch_dB2 = 0, 0
            batch = X[j*30 : (j*30)+30]
            with Pool(processes=30) as pool:
                pool.map(processSequence, batch)

            W1r -= alpha * batch_dW1r / batch_size
            W1h -= alpha * batch_dW1h / batch_size
            B1 -= alpha * batch_dB1 / batch_size
            W2 -= alpha * batch_dW2 / batch_size
            B2 -= alpha * batch_dB2 / batch_size
    return W1r, W1h, B1, W2, B2
















