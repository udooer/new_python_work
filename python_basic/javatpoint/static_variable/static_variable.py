class test:
	count = 0
	def __init__(self):
		test.count += 1
		self.count = 0

if __name__ == "__main__":
	s1 = test();
	print("s1 count:", s1.count)
	print("class count", s1.__class__.count, '\n')
	s2 = test();
	print("s2 count:", s2.count)
	print("s2 class count", s2.__class__.count, '\n')
	s3 = test();
	print("s1 count:", s1.count)
	print("s1 class count", s1.__class__.count, '\n')
	print("test class count:", test.count)

