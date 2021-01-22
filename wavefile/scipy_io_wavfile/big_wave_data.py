#!/usr/bin/python3
import soundfile as sf 
import numpy as np 

data = np.array([[],[]], dtype="float64")
print(data.shape)
# data = np.array([[1],[1]], dtype="float64")
# print(data.shape)
for i in range(10):
    data = np.concatenate( (data, np.array([[i],[i]])), axis=1)

for i in data:
    print(i)