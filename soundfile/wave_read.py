#!/usr/bin/python3
import soundfile as sf 
data, fs = sf.read("10MG.wav")
print("fs is ", fs)
print("data type is", data.dtype)
print("data shape is ", data.shape)
print("data length is ", len(data))
for i in data[:10]:
    print(i)
    print("")