import numpy as np 

def main():
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    b = np.nonzero(a < 5)
    print(b)
    print(a[b])

if __name__ == "__main__":
    main()