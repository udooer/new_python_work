import numpy as np

def main():
    a = np.arange(7,16,1)
    a2 = a[np.newaxis,:]
    print("origin Ndarray:")
    print(a)
    print(a.shape)
    print(a.size)
    print("\nafter adding new axis:")
    print(a2)
    print(a2.shape)
    print(a2.size)
    a3 = np.linspace(0,10,endpoint=False, num=11)
    print(a3)

if __name__ == "__main__":
    main()