class Animal:
    def i(self):
        print("id is animal")
    def bark(self):
        print("Animal bark!!!")
class dog(Animal):
    def bark(self):
        print("Woo Woo")
if __name__ == "__main__":
    d = dog()
    d.i()
    d.bark()