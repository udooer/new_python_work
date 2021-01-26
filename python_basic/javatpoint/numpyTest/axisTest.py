import numpy as np 

def main():
    a = np.arange(15, dtype=np.int).reshape(3,5)
    print(a.dtype)
    Max = np.max(a, axis=0)
    print("a:\n", a)
    print("max value:\n", Max)
    print("\narray item size :\n", a.itemsize, "bytes\n")
    print("try 3D array:")
    a_3D = np.arange(8).reshape(2,2,2)
    Sum = np.sum(a_3D, axis=0)
    print("3D array:", a_3D)
    print("sum of the 3d array along axis 0\n", Sum)

if __name__ == "__main__":
    main()