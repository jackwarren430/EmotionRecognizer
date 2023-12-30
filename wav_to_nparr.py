import librosa
import numpy as np
import os

fixed_length = 115 - 30
count = 0
y_train = np.zeros(1200)
x_train = np.zeros((1200, fixed_length, 13))
for i in range(20):
    if i + 1 < 10:
        folder = "Actor_0" + str(i+1)
    else:
        folder = "Actor_" + str(i+1)
    file_path = "data/archive/" + folder
    file_list = sorted(os.listdir(file_path))
    for j in range(60):
        emotion_class = int(file_list[j][6:8])
        y_train[count] = emotion_class
        file_path_load = file_path + "/" + file_list[j]
        audio, sr = librosa.load(file_path_load, sr=16000) 
        #print(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).shape)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)[:, 40:]

        if mfccs.shape[1] < fixed_length:
            pad_width = fixed_length - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :fixed_length]
        #mfccs_normalized = mfccs / np.amax(mfccs)
        
        mean = np.mean(mfccs, axis=0)
        std_dev = np.std(mfccs, axis=0)
        std_dev[std_dev == 0] = 1
        mfccs = (mfccs - mean) / std_dev
        
        x_train[count] = mfccs.T
        count = count + 1

        
'''
mean = np.mean(x_train, axis=(0,1))
std_dev = np.std(x_train, axis=(0,1))
std_dev[std_dev == 0] = 1
x_train = (x_train - mean) / std_dev
'''

np.random.shuffle(x_train)
np.save("data/raw/x_train.npy", x_train)
np.save("data/raw/y_train.npy", y_train)
