#!/usr/bin/python3
import soundfile as sf 
data, fs = sf.read("forest.wav")
print(type(data))
print(len(data))
for i in data[:10]:
	print(type(i))
	print(i)
