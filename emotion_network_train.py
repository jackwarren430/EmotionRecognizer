from multiprocessing.pool import ThreadPool as Pool
import numpy as np
from functools import partial

x_train = np.load("data/raw/x_train.npy", allow_pickle=True)
y_train = np.load("data/raw/y_train.npy", allow_pickle=True)

s_length = 90 - 20

amount_correct = 0
amount_tested = 1

input_layer_height = 13
middle_layer_height = 10
output_layer_height = 8

def initParams():
    xl = np.sqrt(6 / (middle_layer_height + input_layer_height))
    hl = np.sqrt(6 / (middle_layer_height + middle_layer_height))
    ol = np.sqrt(6 / (output_layer_height + middle_layer_height))
    Wxz = np.random.uniform(-xl, xl, size=(middle_layer_height, input_layer_height))
    Whz = np.random.uniform(-hl, hl, size=(middle_layer_height, middle_layer_height))
    Bz = np.zeros((middle_layer_height, 1))
    Wxr = np.random.uniform(-xl, xl, size=(middle_layer_height, input_layer_height))
    Whr = np.random.uniform(-hl, hl, size=(middle_layer_height, middle_layer_height))
    Br = np.zeros((middle_layer_height, 1))
    Wxh = np.random.uniform(-xl, xl, size=(middle_layer_height, input_layer_height))
    Whh = np.random.uniform(-hl, hl, size=(middle_layer_height, middle_layer_height))
    Bh = np.zeros((middle_layer_height, 1))
    Wo = np.random.uniform(-ol, ol, size=(output_layer_height, middle_layer_height))
    Bo = np.zeros((output_layer_height, 1))
    return [Wxz, Whz, Bz, Wxr, Whr, Br, Wxh, Whh, Bh, Wo, Bo]

def sigmoid(v):
    return 1 / (1 + np.exp(-v))

def sigmoid_prime(v):
    sv = sigmoid(v)
    return sv * (1 - sv)

def tanh(Z):
    return np.tanh(Z)

def tanh_prime(Z):
    return 1 - np.tanh(Z)**2

def softmax(Z):
    e_Z = np.exp(Z - np.max(Z))
    return e_Z / np.sum(e_Z, 0)

def forwardProp(weights_biases, X):
    Wxz, Whz, Bz, Wxr, Whr, Br, Wxh, Whh, Bh, Wo, Bo = weights_biases
    Z_layers = np.zeros((s_length, middle_layer_height, 1))
    R_layers = np.zeros((s_length, middle_layer_height, 1))
    Hhat_layers = np.zeros((s_length, middle_layer_height, 1))
    H_layers = np.zeros((s_length, middle_layer_height, 1))
    A_layers = np.zeros((s_length, output_layer_height, 1))
    H = np.zeros((middle_layer_height, 1))
    for t in range(s_length):
        Uz = Wxz @ X[t].reshape(input_layer_height, 1) + Whz @ H.reshape(middle_layer_height, 1) + Bz.reshape(middle_layer_height, 1)
        Z = sigmoid(Uz)
        Ur = Wxr @ X[t].reshape(input_layer_height, 1) + Whr @ H.reshape(middle_layer_height, 1) + Br.reshape(middle_layer_height, 1)
        R = sigmoid(Ur)
        Uh = Wxh @ X[t].reshape(input_layer_height, 1) + Whh @ (R * H).reshape(middle_layer_height, 1) + Bh.reshape(middle_layer_height, 1)
        Hhat = tanh(Uh)
        Z2 = Wo @ H + Bo
        A = softmax(Z2)
        H = ((1 - Z) * Hhat) + (Z * H)
        Z_layers[t] = Z
        R_layers[t] = R
        Hhat_layers[t] = Hhat 
        H_layers[t] = H
        A_layers[t] = A
    return Z_layers, R_layers, Hhat_layers, H_layers, A_layers


def backwardProp(forward_results, weights_biases, y, X):
    Y = np.zeros((output_layer_height,1))
    Y[int(y-1)][0] = 1
    Z_layers, R_layers, Hhat_layers, H_layers, A_layers = forward_results
    Wxz, Whz, Bz, Wxr, Whr, Br, Wxh, Whh, Bh, Wo, Bo = weights_biases
    dWxz, dWxr, dWxh = (np.zeros((middle_layer_height, input_layer_height)), np.zeros((middle_layer_height, input_layer_height)), np.zeros((middle_layer_height, input_layer_height)))
    dWhz, dWhr, dWhh = (np.zeros((middle_layer_height, middle_layer_height)), np.zeros((middle_layer_height, middle_layer_height)), np.zeros((middle_layer_height, middle_layer_height)))
    dWo = np.zeros((output_layer_height, middle_layer_height))
    dBz, dBr, dBh = (np.zeros((middle_layer_height, 1)), np.zeros((middle_layer_height, 1)), np.zeros((middle_layer_height, 1)))
    dBo = np.zeros((output_layer_height, 1))
    for t in reversed(range(s_length)):
        dL_dZ2 = A_layers[t] - Y
        dBo += dL_dZ2
        dWo += dL_dZ2 @ H_layers[t].T
        dL_dh = Wo.T @ dL_dZ2
        d0 = dL_dh

        Xt = X[t]
        H_prev = H_layers[t - 1] if t > 0 else np.zeros((middle_layer_height, 1))
        Zt = Z_layers[t]
        Hhat = Hhat_layers[t]
        Rt = R_layers[t]

        
        d1 = Zt * d0
        d2 = H_prev * d0 
        d3 = Hhat * d0
        d4 = -1 * d3
        d5 = d2 + d4
        d6 = (1 - Zt) * d0
        d7 = d5 * (Zt * (1 - Zt))
        doutput_layer_height = d6 * (1 - Hhat**2)
        #d9 = Wxh.T @ doutput_layer_height
        dmiddle_layer_height = Whh.T @ doutput_layer_height
        #d11 = Wxz.T @ d7
        #d12 = Whz.T @ d7
        #d14 = dmiddle_layer_height * Rt
        d15 = dmiddle_layer_height * H_prev
        d16 = d15 * (Rt * (1 - Rt))
        #dinput_layer_height = Wxr.T @ d16
        #d17 = Whr.T @ d16

        dWxr += d16 @ Xt.reshape(1, input_layer_height)
        dWxz += d7 @ Xt.reshape(1, input_layer_height)
        dWxh += doutput_layer_height @ Xt.reshape(1, input_layer_height)

        dWhr += d16 @ H_prev.reshape(1, middle_layer_height)
        dWhz += d7 @ H_prev.reshape(1, middle_layer_height)
        dWhh += doutput_layer_height @ (H_prev * Rt).reshape(1, middle_layer_height)

        dBr += d16
        dBz += d7
        dBh += doutput_layer_height


    '''
    gradients = [dWxz, dWhz, dWxr, dWhr, dWxh, dWhh, dWo]
    threshold = 3
    for i in range(7):
        norm = np.linalg.norm(gradients[i], ord=2)
        if norm > threshold:
            gradients[i] *= threshold / norm
    gradients.insert(7, dBo)
    gradients.insert(6, dBh)
    gradients.insert(4, dBr)
    gradients.insert(2, dBz)
    

    dBz /= input_layer_height
    dBr /= input_layer_height
    Bh /= input_layer_height
    dBo /= input_layer_height
    '''

    gradients = [dWxz, dWhz, dBz, dWxr, dWhr, dBr, dWxh, dWhh, dBh, dWo, dBo]
    threshold = 1
    for i in range(11):
        norm = np.linalg.norm(gradients[i], ord=2)
        if norm > threshold:
            gradients[i] *= threshold / norm
    
    
    return gradients
    

def updateParams(weights_biases, gradients, batch_size, alpha):
    for i in range(11):
        d = np.array([item[i] for item in gradients])
        weights_biases[i] -= alpha * np.sum(d, 0) / batch_size
    return weights_biases

def updateParamsLinear(weights_biases, gradients, batch_size, alpha):
    for i in range(11):
        weights_biases[i] -= alpha * gradients[i]
    return weights_biases

def processSequence(sequence, weights_biases):
    global amount_correct, amount_tested
    X, y = sequence
    forward_results = forwardProp(weights_biases, X)
    gradients = backwardProp(forward_results, weights_biases, y, X)
    if (np.argmax(forward_results[-1][-1]) == y):
        amount_correct += 1
        amount_tested += 1
    return gradients

def processSequenceLinear(X, y, weights_biases):
    global amount_correct, amount_tested
    forward_results = forwardProp(weights_biases, X)
    gradients = backwardProp(forward_results, weights_biases, y, X)
    for i in range(forward_results[-1].shape[0]):
        if (np.argmax(forward_results[-1][i]) == y):
            amount_correct += 1
        amount_tested += 1
    return gradients


def gradientDescent(X, Y, iterations, alpha):
    global amount_correct, amount_tested
    weights_biases = initParams()
    batch_size = 6
    for i in range(iterations):
        for j in range(int(X.shape[0]/batch_size)):
            x_batch = X[j*batch_size : (j*batch_size)+batch_size]
            y_batch = Y[j*batch_size : (j*batch_size)+batch_size]
            partial_function = partial(processSequence, weights_biases=weights_biases)
            amount_correct = 0
            with Pool(processes=batch_size) as pool:
                gradients = pool.map(partial_function, zip(x_batch, y_batch))
            weights_biases = updateParams(weights_biases, gradients, batch_size, alpha)
            print("\n ITERATION: {}\n".format(i + 1))
            print("\namount_correct:{}".format(amount_tested))
            print("accuracy: {}".format(amount_correct/batch_size))
    return weights_biases



def gradientDescentLinear(X, Y, iterations, alpha):
    global amount_correct, amount_tested
    weights_biases = initParams()
    batch_size = 6
    for i in range(iterations):
        print("\n ITERATION: {}\n".format(i + 1))
        for j in range(X.shape[0]):
            gradients = processSequenceLinear(X[j], Y[j], weights_biases)
            weights_biases = updateParamsLinear(weights_biases, gradients, batch_size, alpha)
            if (j % 30 == 0):
                print("accuracy: {}".format(10 * amount_correct/amount_tested))
        amount_correct = 0
        amount_tested = 1
        #for i in range(11):
        #   print(weights_biases[i])
        shuffled_indices = np.arange(len(x_train))
        np.random.shuffle(shuffled_indices)
        X = X[shuffled_indices]
        Y = Y[shuffled_indices]
    return weights_biases


#processSequence([x_train[0], y_train[0]], initParams())
#Wxz, Whz, Bz, Wxr, Whr, Br, Wxh, Whh, Bh, Wo, Bo = gradientDescent(x_train, y_train, middle_layer_height00, 0.01)
Wxz, Whz, Bz, Wxr, Whr, Br, Wxh, Whh, Bh, Wo, Bo = gradientDescentLinear(x_train, y_train, 1000, 0.001)

np.save("./data/weightsBiases/Wxz.npy", Wxz)
np.save("./data/weightsBiases/Whz.npy", Whz)
np.save("./data/weightsBiases/Bz.npy", Bz)
np.save("./data/weightsBiases/Wxr.npy", Wxr)
np.save("./data/weightsBiases/Whr.npy", Whr)
np.save("./data/weightsBiases/Br.npy", Br)
np.save("./data/weightsBiases/Wxh.npy", Wxh)
np.save("./data/weightsBiases/Whh.npy", Whh)
np.save("./data/weightsBiases/Bh.npy", Bh)
np.save("./data/weightsBiases/Wo.npy", Wo)
np.save("./data/weightsBiases/Bo.npy", Bo)











