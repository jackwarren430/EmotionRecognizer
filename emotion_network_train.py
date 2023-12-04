from multiprocessing.pool import ThreadPool as Pool
import numpy as np
from functools import partial

x_train = np.load("data/raw/x_train.npy")
y_train = np.load("data/raw/y_train.npy")

amount_correct = 0
amount_tested = 0

def initParams():
    W1r = np.random.randn(10, 13) * np.sqrt(2. / 13)
    W1h = np.random.randn(10, 10) * np.sqrt(2. / 10)
    B1 = np.zeros((10, 1))
    W2 = np.random.randn(8, 10) * np.sqrt(2. / 10)
    B2 = np.zeros((8, 1))
    return W1r, W1h, B1, W2, B2

def relu(Z):
    return np.maximum(0, Z)

def relu_prime(Z):
    return Z > 0

def softmax(Z):
    #fix
    shift_z = Z - np.max(Z, 0)
    exp_scores = np.exp(shift_z)
    probabilities = exp_scores / np.sum(exp_scores, 0)
    return probabilities

def forwardProp(W1r, W1h, B1, W2, B2, X):
    hidden_states = np.zeros((216, 10))
    hidden_states_u = np.zeros((216, 10))
    hidden_layer = np.zeros((10, 1))
    outputs = np.zeros((216, 8))
    for i in range(216):
        hidden_layer = W1r.dot(X[i]) + W1h.dot(hidden_layer).reshape(-1) + B1.reshape(-1)
        #print("\nt: {}, hidden: {}\n".format(i, hidden_layer))
        hidden_states_u[i] = hidden_layer
        hidden_states[i] = relu(hidden_layer)
        Z2 = W2.dot(relu(hidden_layer)).reshape((8, 1)) + B2.reshape((8, 1))
        A2 = softmax(Z2)
        outputs[i] = A2.reshape(-1)
    return hidden_states, hidden_states_u, outputs, Z2, A2

def backwardProp(hidden_states, hidden_states_u, outputs, A2, W2, y, X):
    Y = np.zeros((8,1))
    Y[int(y-1)][0] = 1
    dW1r = np.zeros((10, 13))
    dW1h = np.zeros((10, 10))
    dB1 = np.zeros((10, 1))
    dW2 = np.zeros((8, 10))
    dB2 = np.zeros((8, 1))
    #test updating the gradient each time with each output - t1       
    #gradient = A2 - Y
    #test with just this or accumilating - t2
    #dB2 = gradient / 13
    #dW2 = gradient.dot(hidden_states[215].reshape((1, 10)))
    dh_dht_1 = 1
    for t in reversed(range(216)):
        gradient = outputs[t] - Y
        threshold = 5
        gradient_norm = np.linalg.norm(gradient, ord=2)
        if gradient_norm > threshold:
            gradient = (gradient / gradient_norm) * threshold
        
        #dy/dh
        dy_dh = 
        #dh_t/dh_t-1
        dht = 
        dh_dht_1 = dh_dht_1.dot(dht)
        #dh_t-1/dWxh
        dht1_dW1r = 
        dht1_dW1h = 

        dW1r_t = gradient.dot(dy_dh.dot(dh_dht_1.dot(dht1_dW1r)))
        dW1h_t = gradient.dot(dy_dh.dot(dh_dht_1.dot(dht1_dW1h)))

        

        dW2 += dW2_t
        dB2 += dB2_t
        dW1r += dW1r_t
        dW1h += dW1h_t
        dB1 += dB1_t

    return dW1r, dW1h, dB1, dW2, dB2

def updateParams(W1r, W1h, B1, W2, B2, results, batch_size, alpha):
    dW1r = np.array([item[0] for item in results])
    dW1h = np.array([item[1] for item in results])
    dB1 = np.array([item[2] for item in results])
    dW2 = np.array([item[3] for item in results])
    dB2 = np.array([item[4] for item in results])
    W1r -= alpha * np.sum(dW1r, 0) / batch_size
    W1h -= alpha * np.sum(dW1h, 0) / batch_size
    B1 -= alpha * np.sum(dB1, 0) / batch_size
    W2 -= alpha * np.sum(dW2, 0) / batch_size
    B2 -= alpha * np.sum(dB2, 0) / batch_size
    return W1r, W1h, B1, W2, B2

def processSequence(x, y, weights_biases):
    global amount_correct, amount_tested
    #fix back to partial function
    #x, y = sequence
    x = (x - np.mean(x)) / np.std(x)
    print(np.ptp(x))
    W1r, W1h, B1, W2, B2 = weights_biases
    hidden_states, hidden_states_u, outputs, Z2, A2 = forwardProp(W1r, W1h, B1, W2, B2, x)
    dW1r, dW1h, dB1, dW2, dB2 = backwardProp(hidden_states, hidden_states_u, outputs, A2, W2, y, x)
    if (np.argmax(A2)==y):
        amount_correct += 1
    amount_tested += 1
    return [dW1r, dW1h, dB1, dW2, dB2]

def gradient_descent(X, Y, iterations, alpha):
    W1r, W1h, B1, W2, B2 = initParams()
    batch_size = 30
    for i in range(iterations):
        #for j in range(int(300/batch_size)):
        for j in range(1):
            x_batch = X[j*batch_size : (j*batch_size)+batch_size]
            y_batch = Y[j*batch_size : (j*batch_size)+batch_size]
            weights_biases = [W1r, W1h, B1, W2, B2]

            #partial_function = partial(processSequence, weights_biases=weights_biases)
            #with Pool(processes=batch_size) as pool:
                #results = pool.map(partial_function, zip(x_batch, y_batch))

            #test
            results = processSequence(X[0], Y[0], weights_biases)
            
            W1r, W1h, B1, W2, B2 = updateParams(W1r, W1h, B1, W2, B2, results, batch_size, alpha)
        #print("iteration: {}, accuracy: {}".format(i, amount_correct/amount_tested))
    return W1r, W1h, B1, W2, B2

gradient_descent(x_train, y_train, 1, 0.01)














