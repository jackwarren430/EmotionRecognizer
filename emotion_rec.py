import numpy as np

x_train = np.load("data/raw/x_train.npy")
y_train = np.load("data/raw/y_train.npy")
n, m = x_train.shape

def initParams():
    W1 = np.random.randn(8, 20)
    B1 = np.random.randn(8, 1)
    W2 = np.random.randn(8, 8)
    B2 = np.random.randn(8, 1)
    return W1, B1, W2, B2

def relu(Z):
    return np.maximum(0, Z)

def relu_prime(Z):
    return Z > 0

def softmax(Z):
    shift_z = Z - np.max(Z, 0)
    exp_scores = np.exp(shift_z)
    probabilities = exp_scores / np.sum(exp_scores, 0)
    return probabilities

def getY(y_train):
    to_testY = np.zeros((8, 1440))
    for i in range(1440):
        index = int(y_train[i]) - 1
        to_testY[index][i] = 1
    return to_testY

def forwardProp(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1
    A1 = relu(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backProp(Z1, A1, Z2, A2, W2, X, Y):
    Y = getY(Y)
    dZ2 = A2 - Y
    dW2 = (dZ2.dot(A1.T))/m
    dB2 = (np.sum(dZ2, 1).reshape(-1, 1))/m
    dZ1 = W2.T.dot(dZ2) * relu_prime(Z1)
    dW1 = (dZ1.dot(X.T))/m
    dB1 = (np.sum(dZ1, 1).reshape(-1, 1))/m
    return dW1, dB1, dW2, dB2

def updateParams(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha):
    W1 = W1 - alpha * dW1
    B1 = B1 - alpha * dB1
    W2 = W2 - alpha * dW2
    B2 = B2 - alpha * dB2
    return W1, B1, W2, B2

def get_predictions(A2):
    print(A2[:, 0])
    print(A2[:, 1])
    print(np.argmax(A2, 0)[0:10])
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, B1, W2, B2 = initParams()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forwardProp(W1, B1, W2, B2, X)
        dW1, dB1, dW2, dB2 = backProp(Z1, A1, Z2, A2, W2, X, Y)
        W1, B1, W2, B2 = updateParams(W1, B1, W2, B2, dW1, dB1, dW2, dB2, alpha)
        if (i % 50 == 0):
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
    return W1, B1, W2, B2

print(x_train[:, 0])
print(x_train[:, 1])
print(x_train[:, 2])
input()
gradient_descent(x_train, y_train, 10000, 0.01)



