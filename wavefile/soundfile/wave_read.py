#!/usr/bin/python3
import soundfile as sf 
data, fs = sf.read("write.wav")
print(type(data))
print(len(data))
for i in data:
	a = i*pow(2,31);
	print(a)
