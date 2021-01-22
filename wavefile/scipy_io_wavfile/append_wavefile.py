#!/usr/bin/python3
import scipy.io.wavfile as wavfile
import numpy as np
fs = 44100
data = np.array([[],[]], dtype="float32")
print("data shape is ", data.shape, ", data type is ", data.dtype)
wavfile.write("example.wav", fs, data)

data = np.array([[1],[1]], dtype="float32")
wavfile.write("example.wav", fs, data)

data = np.array([[1,2,3],[1,2,3]], dtype="float32")
wavfile.write("example.wav", fs, data)

fs, data_get = wavfile.read("example.wav")

print("data size we get ", len(data_get[0]))
for i in data_get:
	print(i)
