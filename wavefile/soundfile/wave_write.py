#!/usr/bin/python3
import soundfile as sf 
import numpy as np 
data = np.array([[0,0]], dtype="float32")
for i in range(1, 200000):
	data = np.concatenate((data, np.array([[i, -i]], dtype="float32")), axis=0)
print(data.shape)
print(data.dtype)
print(data)
sf.write("test.wav", data, 96000)