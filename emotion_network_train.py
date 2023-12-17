from multiprocessing.pool import ThreadPool as Pool
import numpy as np
from functools import partial

x_train = np.load("data/raw/x_train.npy")
y_train = np.load("data/raw/y_train.npy")

mean = np.mean(x_train, axis=(0, 1))
std_dev = np.std(x_train, axis=(0, 1))
std_dev[std_dev == 0] = 1
x_train = (x_train - mean) / std_dev

amount_correct = 0
amount_tested = 0

def initParams():
    xl = np.sqrt(6 / (10 + 13))
    hl = np.sqrt(6 / (10 + 10))
    ol = np.sqrt(6 / (8 + 10))
    np.random.uniform(-xl, xl, size=(10, 13))
    Wxz = np.random.uniform(-xl, xl, size=(10, 13))
    Whz = np.random.uniform(-hl, hl, size=(10, 10))
    Bz = np.zeros((10, 1))
    Wxr = np.random.uniform(-xl, xl, size=(10, 13))
    Whr = np.random.uniform(-hl, hl, size=(10, 10))
    Br = np.zeros((10, 1))
    Wxh = np.random.uniform(-xl, xl, size=(10, 13))
    Whh = np.random.uniform(-hl, hl, size=(10, 10))
    Bh = np.zeros((10, 1))
    Wo = np.random.uniform(-ol, ol, size=(8, 10))
    Bo = np.zeros((8, 1))
    return Wxz, Whz, Bz, Wxr, Whr, Br, Wxh, Whh, Bh, Wo, Bo

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
    Uz_layers = np.zeros((216, 10))
    Z_layers = np.zeros((216, 10))
    Ur_layers = np.zeros((216, 10))
    R_layers = np.zeros((216, 10))
    Uh_layers = np.zeros((216, 10))
    Hhat_layers = np.zeros((216, 10))
    H_layers = np.zeros((216, 10))
    H = np.zeros((10, 1))
    for t in range(216):
        Uz = Wxz @ X[t] + Whz @ H + Bz
        Z = sigmoid(Uz)
        Ur = Wxr @ X[t] + Whr @ H + Br
        R = sigmoid(Ur)
        Uh = Wxh @ X[t] + Whh @ (R * H) + Bh
        Hhat = tanh(Uh)
        H = (1 - Z) * H + Z * Hhat
        Uz_layers[t] = Uz
        Z_layers[t] = Z
        Ur_layers[t] = Ur
        R_layers[t] = R
        Uh_layers[t] = Uh
        Hhat_layers[t] = Hhat 
        H_layers[t] = H
    Z2 = Wo @ H + Bo
    A = softmax(Z2)
    return Uz_layers, Z_layers, Ur_layers, R_layers, Uh_layers, Hhat_layers, Hhat_layers, Z2, A

def calculateGradients(foward_results, weights_biases, t, X):
    Uz_layers, Z_layers, Ur_layers, R_layers, Uh_layers, Hhat_layers, Hhat_layers, Z2, A = foward_results
    Wxz, Whz, Bz, Wxr, Whr, Br, Wxh, Whh, Bh, Wo, Bo = weights_biases
    if (t == 0):
        #base case
        return
    dh_Wxz, dh_Whz, dh_Bz, dh_Wxr, dh_Whr, dh_Br, dh_Wxh, dh_Whh, dh_Bh, dh_Wo, dh_Bo = calculateGradients(foward_results, weights_biases, t)
    dh_Wxz = (sigmoid_prime(Uz_layers[t]) @ (X[t] + Whz @ ))
    dh_Whz
    dh_Bz
    dh_Wxr
    dh_Whr
    dh_Br
    dh_Wxh
    dh_Whh
    dh_Bh
    dh_Wo
    dh_Bo
    return dh_Wxz, dh_Whz, dh_Bz, dh_Wxr, dh_Whr, dh_Br, dh_Wxh, dh_Whh, dh_Bh, dh_Wo, dh_Bo

def backwardProp(foward_results, weights_biases, y, X):
    Y = np.zeros((8,1))
    Y[int(y-1)][0] = 1
    
    t = 216
    dh_Wxz, dh_Whz, dh_Bz, dh_Wxr, dh_Whr, dh_Br, dh_Wxh, dh_Whh, dh_Bh, dh_Wo, dh_Bo = calculateGradients(foward_results, weights_biases, t, X)
    dWxz
    dWhz
    dBz
    dWxr
    dWhr
    dBr
    dWxh
    dWhh
    dBh
    dWo
    dBo

   
    '''
    Wx_threshold = 5
    dWx_norm = np.linalg.norm(dWx, ord=2)
    if dWx_norm > Wx_threshold:
        dWx *= Wx_threshold / dWx_norm
    '''
    return dWxz, dWhz, dBz, dWxr, dWhr, dBr, dWxh, dWhh, dBh, dWo, dBo

def updateParams(Wx, Wh, B1, Wo, B2, results, batch_size, alpha):
    dWx = np.array([item[0] for item in results])
    dWh = np.array([item[1] for item in results])
    dB1 = np.array([item[2] for item in results])
    dWo = np.array([item[3] for item in results])
    dB2 = np.array([item[4] for item in results])
    hold = "\n\n-----------\n\ndWx:\n{}\n---\ndWh:\n{}\n---\ndB1:\n{}\n\n\n\n-----------".format(np.sum(dWx, 0), np.sum(dWh, 0), np.sum(dB1, 0))
    Wx -= alpha * np.sum(dWx, 0) / batch_size
    Wh -= alpha * np.sum(dWh, 0) / batch_size
    B1 -= alpha * np.sum(dB1, 0) / batch_size
    Wo -= alpha * np.sum(dWo, 0) / batch_size
    B2 -= alpha * np.sum(dB2, 0) / batch_size
    hold2 = "\n\n-----------\n\nWx_updated:\n{}\n---\ndWh_updated:\n{}\n---\ndB1_updated:\n{}\n\n\n\n-----------".format(Wx, Wh, B1)
    #print(hold + hold2)

    return Wx, Wh, B1, Wo, B2


#def processSequence(x, y, weights_biases):
def processSequence(sequence, weights_biases):
    global amount_correct, amount_tested
    X, y = sequence
    foward_results = forwardProp(weights_biases, X)
    gradients = backwardProp(foward_results, y, X)
    if (np.argmax(A) == y):
        amount_correct += 1
    amount_tested += 1
    return gradients

def gradient_descent(X, Y, iterations, alpha):
    Wxz, Whz, Bz, Wxr, Whr, Br, Wxh, Whh, Bh, Wo, Bo = initParams()
    batch_size = 6
    for i in range(iterations):
        print("\n ITERATION: {}\n".format(i + 1))
        #for j in range(300):
        for j in range(int(300/batch_size)):
            #print("\n  BATCH: {}\n".format(j + 1))
            x_batch = X[j*batch_size : (j*batch_size)+batch_size]
            y_batch = Y[j*batch_size : (j*batch_size)+batch_size]
            weights_biases = [Wxz, Whz, Bz, Wxr, Whr, Br, Wxh, Whh, Bh, Wo, Bo]

            
            partial_function = partial(processSequence, weights_biases=weights_biases)
            with Pool(processes=batch_size) as pool:
                gradients = pool.map(partial_function, zip(x_batch, y_batch))
            Wxz, Whz, Bz, Wxr, Whr, Br, Wxh, Whh, Bh, Wo, Bo = updateParams(weights_biases, gradients, batch_size, alpha)
            

            #processSequence(X[j], Y[j], weights_biases)
            print("\namount_correct:{}".format(amount_correct))
            print("accuracy: {}".format(amount_correct/amount_tested))
    return Wx, Wh, B1, Wo, B2

Wx, Wh, B1, Wo, B2 = gradient_descent(x_train, y_train, 1000, 0.2)

np.save("./data/weightsBiases/Wx.npy", Wx)
np.save("./data/weightsBiases/Wh.npy", Wh)
np.save("./data/weightsBiases/B1.npy", B1)
np.save("./data/weightsBiases/Wo.npy", Wo)
np.save("./data/weightsBiases/B2.npy", B2)












