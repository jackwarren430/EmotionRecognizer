import librosa
import numpy as np
import os

Wxz = np.load("./data/weightsBiases/Wxz.npy")
Whz = np.load("./data/weightsBiases/Whz.npy")
Bz = np.load("./data/weightsBiases/Bz.npy") 
Wxr = np.load("./data/weightsBiases/Wxr.npy")
Whr = np.load("./data/weightsBiases/Whr.npy")
Br = np.load("./data/weightsBiases/Br.npy")
Wxh = np.load("./data/weightsBiases/Wxh.npy")
Whh = np.load("./data/weightsBiases/Whh.npy")
Bh =  np.load("./data/weightsBiases/Bh.npy")
Wo = np.load("./data/weightsBiases/Wo.npy")
Bo = np.load("./data/weightsBiases/Bo.npy") 


def sigmoid(v):
    return 1 / (1 + np.exp(-v))

def tanh(Z):
    return np.tanh(Z)

def softmax(Z):
    e_Z = np.exp(Z - np.max(Z))
    return e_Z / np.sum(e_Z, 0)

def forwardProp(weights_biases, X):
    A_layers = np.zeros((x.shape[0], 8, 1))
    H = np.zeros((10, 1))
    for t in range(x.shape[0]):
        Uz = Wxz @ X[t].reshape(13, 1) + Whz @ H.reshape(10, 1) + Bz.reshape(10, 1)
        Z = sigmoid(Uz)
        Ur = Wxr @ X[t].reshape(13, 1) + Whr @ H.reshape(10, 1) + Br.reshape(10, 1)
        R = sigmoid(Ur)
        Uh = Wxh @ X[t].reshape(13, 1) + Whh @ (R * H).reshape(10, 1) + Bh.reshape(10, 1)
        Hhat = tanh(Uh)
        Z2 = Wo @ H + Bo
        A = softmax(Z2)
        H = ((1 - Z) * Hhat) + (Z * H)
        A_layers[t] = A
    return A_layers

def prepSequence(actor, emotion):
	file_path = "data/archive/Actor_{}/03-01-0{}-01-01-01-{}.wav".format(actor, emotion, actor)
	audio, sr = librosa.load(file_path, sr=16000)
	mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)[:, 20:90]
	mean = np.mean(mfccs, axis=0)
	std_dev = np.std(mfccs, axis=0)
	std_dev[std_dev == 0] = 1
	mfccs = (mfccs - mean) / std_dev
	return mfccs.T


actor = "11"
emotion = "5"

weights_biases = Wxz, Whz, Bz, Wxr, Whr, Br, Wxh, Whh, Bh, Wo, Bo
x = prepSequence(actor, emotion)

results = forwardProp(weights_biases, x)

for i in range(results.shape[0]):
	print(np.argmax(results[i]))

print("Emotion was: {}".format(emotion))

