import numpy as np 

def main():
	a = np.arange(10).reshape(5,2)
	b = np.arange(5)+100

	c = np.concatenate((a,b[:,np.newaxis]), axis=1)
	print("a:\n", a)
	print("b:\n", b)
	print("\nc:\n", c)

if __name__ == "__main__":
	main()
