import numpy as np
def main():
	a = [1,2,3]
	b = [4,5,6]
	c = list(zip(a,b))
	print(c)
	d = list(zip(*c))
	print(d)

if __name__ == "__main__":
	main()