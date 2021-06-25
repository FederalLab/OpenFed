verbose = True

class A():
    def __enter__(self):
        if not verbose:
            self.__exit__()
    
    def __exit__(self, *args, **kwargs):
        return 

a = A()
with a:
    print("Here")

verbose = False
with a:
    print("Here")