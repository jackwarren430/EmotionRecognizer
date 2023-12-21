from multiprocessing.pool import ThreadPool as Pool
import numpy as np
from functools import partial

x_train = np.load("data/raw/x_train.npy", allow_pickle=True)
y_train = np.load("data/raw/y_train.npy", allow_pickle=True)



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
    Uz_layers = np.zeros((216, 10, 1))
    Z_layers = np.zeros((216, 10, 1))
    Ur_layers = np.zeros((216, 10, 1))
    R_layers = np.zeros((216, 10, 1))
    Uh_layers = np.zeros((216, 10, 1))
    Hhat_layers = np.zeros((216, 10, 1))
    H_layers = np.zeros((216, 10, 1))
    H = np.zeros((10, 1))
    for t in range(216):
        Uz = Wxz @ X[t].reshape(13, 1) + Whz @ H.reshape(10, 1) + Bz.reshape(10, 1)
        Z = sigmoid(Uz)
        Ur = Wxr @ X[t].reshape(13, 1) + Whr @ H.reshape(10, 1) + Br.reshape(10, 1)
        R = sigmoid(Ur)
        Uh = Wxh @ X[t].reshape(13, 1) + Whh @ (R * H).reshape(10, 1) + Bh.reshape(10, 1)
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
    return Uz_layers, Z_layers, Ur_layers, R_layers, Uh_layers, Hhat_layers, H_layers, Z2, A


def backwardProp(forward_results, weights_biases, y, X):
    Y = np.zeros((8,1))
    Y[int(y-1)][0] = 1
    Uz_layers, Z_layers, Ur_layers, R_layers, Uh_layers, Hhat_layers, H_layers, Z2, A = forward_results
    Wxz, Whz, Bz, Wxr, Whr, Br, Wxh, Whh, Bh, Wo, Bo = weights_biases
    dWxz, dWxr, dWxh = (np.zeros((10, 13)), np.zeros((10, 13)), np.zeros((10, 13)))
    dWhz, dWhr, dWhh = (np.zeros((10, 10)), np.zeros((10, 10)), np.zeros((10, 10)))
    dBz, dBr, dBh = (np.zeros((10, 1)), np.zeros((10, 1)), np.zeros((10, 1)))
    dL_dZ2 = A - Y
    #print(Whh)
    #print(dL_dZ2)
    dBo = dL_dZ2
    dWo = dL_dZ2 @ H_layers[-1].T
    dL_dh = Wo.T @ dL_dZ2
    #print(dL_dh)
    for t in reversed(range(216)):
        H_prev = H_layers[t - 1] if t >= 0 else np.zeros(10, 1)

        dh_dz = np.diag((Hhat_layers[t] - H_prev).flatten())
        dz_dWxz = sigmoid_prime(Uz_layers[t]) @ X[t].T.reshape(1, 13)
        dz_dWhz = sigmoid_prime(Uz_layers[t]) @ H_prev.T
        dz_dBz = sigmoid_prime(Uz_layers[t])
        #hold = "******\nt: {}\ndL_dh:\n{}\n\ndh_dz:\n{}\n\ndz_dWxz:\n{}\n\n"
        #print(hold.format(t, dL_dh, dh_dz, dz_dWxz))
        #print(Whr.shape)
        #print((sigmoid_prime(Ur_layers[t]) * Whr).shape)
        dWxz += dL_dh * (dh_dz @ dz_dWxz)
        dWhz += dL_dh * (dh_dz @ dz_dWhz)
        dBz += dL_dh * (dh_dz @ dz_dBz)
        #hold = "*****\nt: {} \ndWxz update:\n{}\n\n".format(t, dL_dh * (dh_dz * dz_dWxz))
        #print(hold)

        dh_dhh = np.diag(Z_layers[t].flatten())
        dhh_dWxh = tanh_prime(Uh_layers[t]) @ X[t].T.reshape(1, 13)
        dhh_dWhh = tanh_prime(Uh_layers[t]) @ (R_layers[t] * H_prev).T
        dhh_dBh = tanh_prime(Uh_layers[t])
        # dL_dh is 10x1
        # dh_dhh is 10x10
        dWxh += dL_dh * (dh_dhh @ dhh_dWxh)
        dWhh += dL_dh * (dh_dhh @ dhh_dWhh)
        dBh += dL_dh * (dh_dhh @ dhh_dBh)

        #dh_dr = np.diag(Z_layers[t].flatten()) @ (tanh_prime(Uh_layers[t]) @ (Whh @ H_prev).T)  # might need to fix -------
        dh_dr = np.diag(Z_layers[t].flatten()) @ ((Whh @ np.diag(H_prev.flatten())) * tanh_prime(Uh_layers[t])) # might need to fix -------
        dr_dWxr = sigmoid_prime(Ur_layers[t]) @ X[t].T.reshape(1, 13)
        dr_dWhr = sigmoid_prime(Ur_layers[t]) @ H_prev.T
        dr_dBr = sigmoid_prime(Ur_layers[t])
        #print(Whh)
        dWxr += dL_dh * (dh_dr @ dr_dWxr)
        dWhr += dL_dh * (dh_dr @ dr_dWhr)
        dBr += dL_dh * (dh_dr @ dr_dBr)
        
        dh_dhprev = np.diag((1 - Z_layers[t]).flatten()) - ((Whz @ sigmoid_prime(Uz_layers[t])) @ (Hhat_layers[t] - H_prev).T) + (Z_layers[t] * ((Whh @ (((Whr @ sigmoid_prime(Ur_layers[t])) * H_prev) + R_layers[t])) * tanh_prime(Uh_layers[t]).T))
        #print(dh_dhprev.shape)
        #hold = "*****\nt: {} \ndL_dh:\n{}\n\ndh_dhprev: \n{}\n\ncombined:\n{}\n\n"
        #print(hold.format(t, dL_dh, dh_dhprev, dL_dh * dh_dhprev))
        #print("*****\nt: {}\ndh_dhprev:\n{}\n-----\n".format(t, dh_dhprev))

        #dh_dhprev is 10x10
        dL_dh = dh_dhprev @ dL_dh
   
    return dWxz, dWhz, dBz, dWxr, dWhr, dBr, dWxh, dWhh, dBh, dWo, dBo

def updateParams(weights_biases, gradients, batch_size, alpha):
    #print(gradients[0][0])
    #print("********")
    for i in range(11):
        d = np.array([item[i] for item in gradients])
        weights_biases[i] -= alpha * np.sum(d, 0) / batch_size
        #print(alpha * np.sum(d, 0) / batch_size)
    return weights_biases


def processSequence(sequence, weights_biases):
    global amount_correct, amount_tested
    X, y = sequence
    forward_results = forwardProp(weights_biases, X)
    gradients = backwardProp(forward_results, weights_biases, y, X)
    if (np.argmax(forward_results[-1]) == y):
        amount_correct += 1
    amount_tested += 1
    return gradients

def gradient_descent(X, Y, iterations, alpha):
    weights_biases = initParams()
    batch_size = 6
    for i in range(iterations):
        print("\n ITERATION: {}\n".format(i + 1))
        for j in range(int(X.shape[0]/batch_size)):
            #print("\n  BATCH: {}\n".format(j + 1))
            x_batch = X[j*batch_size : (j*batch_size)+batch_size]
            y_batch = Y[j*batch_size : (j*batch_size)+batch_size]

            
            partial_function = partial(processSequence, weights_biases=weights_biases)
            with Pool(processes=batch_size) as pool:
                gradients = pool.map(partial_function, zip(x_batch, y_batch))
            weights_biases = updateParams(weights_biases, gradients, batch_size, alpha)

            
            print("\namount_correct:{}".format(amount_correct))
            print("accuracy: {}".format(amount_correct/amount_tested))
    return Wxz, Whz, Bz, Wxr, Whr, Br, Wxh, Whh, Bh, Wo, Bo

Wxz, Whz, Bz, Wxr, Whr, Br, Wxh, Whh, Bh, Wo, Bo = gradient_descent(x_train, y_train, 1, 0.1)

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











