from multiprocessing.pool import ThreadPool as Pool
import numpy as np
from functools import partial

x_train = np.load("data/raw/x_train.npy")
y_train = np.load("data/raw/y_train.npy")


amount_correct = 0
amount_tested = 0

def initParams():
    Wx = np.random.randn(10, 13) * np.sqrt(2. / 13)
    Wh = np.random.randn(10, 10) * np.sqrt(2. / 10)
    B1 = np.zeros((10, 1))
    Wo = np.random.randn(8, 10) * np.sqrt(2. / 10)
    B2 = np.zeros((8, 1))
    return Wx, Wh, B1, Wo, B2

def relu(Z):
    return np.maximum(0, Z)

def relu_prime(Z):
    return Z > 0

def tanh(Z):
    return np.tanh(Z)

def tanh_prime(Z):
    #print("\nnp.tanh(Z):\n{}\ntanhprime:\n{}\n".format(np.tanh(Z), 1 - np.tanh(Z)**2))
    return 1 - np.tanh(Z)**2

def softmax(Z):
    e_Z = np.exp(Z - np.max(Z))
    return e_Z / np.sum(e_Z, 0)

def forwardProp(Wx, Wh, B1, Wo, B2, X):
    hidden_states = np.zeros((216, 10))
    hidden_states_u = np.zeros((216, 10))
    hidden_layer = np.zeros((10, 1))
    for i in range(216):
        #print("$$$$$$$$\n$$$$$$$\nwx . x:\n{}\n\nwh . h:\n{}\n\n$$$$$$$$$\n$$$$$$$$\n".format(Wx.dot(X[i]), Wh.dot(hidden_layer).reshape(-1)))
        hidden_layer = (Wx.dot(X[i]) + Wh.dot(hidden_layer).reshape(-1) + B1.reshape(-1))/2
        hidden_states_u[i] = hidden_layer
        hidden_states[i] = relu(hidden_layer)
    Z = Wo.dot(relu(hidden_layer)).reshape((8, 1)) + B2.reshape((8, 1))
    #print("^^^^^^\nZ:\n{}\n^^^^^^^^".format(Z))
    A = softmax(Z)
    #print("^^^^^^\nA:\n{}\n^^^^^^^^".format(A))
    return hidden_states_u, hidden_states, Z, A

def backwardProp(hidden_states_u, hidden_states, Z, A, Wh, Wo, y, X):
    Y = np.zeros((8,1))
    Y[int(y-1)][0] = 1
    dWx = np.zeros((10, 13))
    dWh = np.zeros((10, 10))
    dB1 = np.zeros((10, 1))
    
    gradient = A - Y
    dB2 = gradient / 13
    dWo = gradient @ hidden_states[-1].reshape((1, 10))
    agg = 1
    for t in reversed(range(216)):
        curr_x = X[t]
        curr_hid_u = hidden_states_u[t]
        prev_hid = hidden_states[t - 1] if t >= 0 else 0

        if (t == 215):
            #print("\n\n\n---\nWo.T:\n{}\n\ncurr_hid_u:\n{}\n---\n\n".format(Wo.T, curr_hid_u))
            agg = (Wo.T @ gradient) * relu_prime(curr_hid_u).reshape((10, 1))
            #print("\n\n\n---\nagg:\n{}\n\ncurr X:\n{}\n---\n\n*******".format(agg, curr_x.T))
        else:
            agg = (Wh @ agg) * relu_prime(curr_hid_u).reshape((10, 1)) 
            #print("\n\ncurr_hid_u:\n{}\n---\n\n".format(curr_hid_u))

        #look into subtracting a value before multiplying, then add them back in
        dWx_t = np.outer(agg, curr_x.T)
        dWh_t = np.outer(agg, prev_hid)
        dB1_t = agg / 13

        dWx += dWx_t
        dWh += dWh_t
        dB1 += dB1_t
    
    hold = "\n\n\n----------\ndWx:\n{}\n---\ndWh:\n{}\n---\ndB1:\n{}\n\n".format(dWx, dWh, dB1)

    Wx_threshold = 5
    dWx_norm = np.linalg.norm(dWx, ord=2)
    if dWx_norm > Wx_threshold:
        dWx *= Wx_threshold / dWx_norm
    Wh_threshold = 5
    dWh_norm = np.linalg.norm(dWh, ord=2)
    if dWh_norm > Wh_threshold:
        dWh *= Wh_threshold / dWh_norm
    B_threshold = 5
    dB1_norm = np.linalg.norm(dB1, ord=2)
    if dB1_norm > B_threshold:
        dB1 *= B_threshold / dB1_norm

    hold2 = "\n---\ndWx_clipped:\n{}\n---\ndWh_clipped:\n{}\n---\ndB1_clipped:\n{}\n\n\n\n-----------".format(dWx, dWh, dB1)
    #print(hold + hold2)

    return dWx, dWh, dB1, dWo, dB2

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
    x, y = sequence
    #x = (x - np.mean(x, 0)) / np.std(x, 0)
    x = 2*((x - np.min(x)) / (np.max(x) - np.min(x))) - 1
    Wx, Wh, B1, Wo, B2 = weights_biases
    hidden_states_u, hidden_states, Z, A = forwardProp(Wx, Wh, B1, Wo, B2, x)
    dWx, dWh, dB1, dWo, dB2 = backwardProp(hidden_states_u, hidden_states, A, Z, Wh, Wo, y, x)
    if (np.argmax(A) == y):
        amount_correct += 1
    amount_tested += 1
    return [dWx, dWh, dB1, dWo, dB2]

def gradient_descent(X, Y, iterations, alpha):
    Wx, Wh, B1, Wo, B2 = initParams()
    batch_size = 6
    for i in range(iterations):
        print("\n ITERATION: {}\n".format(i + 1))
        #for j in range(300):
        for j in range(int(300/batch_size)):
            #print("\n  BATCH: {}\n".format(j + 1))
            x_batch = X[j*batch_size : (j*batch_size)+batch_size]
            y_batch = Y[j*batch_size : (j*batch_size)+batch_size]
            weights_biases = [Wx, Wh, B1, Wo, B2]

            
            partial_function = partial(processSequence, weights_biases=weights_biases)
            with Pool(processes=batch_size) as pool:
                results = pool.map(partial_function, zip(x_batch, y_batch))
            Wx, Wh, B1, Wo, B2 = updateParams(Wx, Wh, B1, Wo, B2, results, batch_size, alpha)
            

            #processSequence(X[j], Y[j], weights_biases)

            print("accuracy: {}".format(amount_correct/amount_tested))
    return Wx, Wh, B1, Wo, B2

Wx, Wh, B1, Wo, B2 = gradient_descent(x_train, y_train, 1000, 0.1)

np.save("./data/weightsBiases/Wx.npy", Wx)
np.save("./data/weightsBiases/Wh.npy", Wh)
np.save("./data/weightsBiases/B1.npy", B1)
np.save("./data/weightsBiases/Wo.npy", Wo)
np.save("./data/weightsBiases/B2.npy", B2)












