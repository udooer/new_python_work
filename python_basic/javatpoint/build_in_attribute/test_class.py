class test:
    def __init__(self):
        self.count = 0
        self.l = [1,2,3,4]
        self.s = "shane"
        print("init function:", __name__)

if __name__ == "__main__":
    t = test()
    print("__dict__:", t.__dict__)
    print("__module__:", t.__module__)