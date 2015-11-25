import networkx as nx
import os
import pz

# test section -------------------------------------------------------------------
class MyClass:
    "A simple example class"
    i = 12345
    def f(self):
        "gar nicht schlecht!"
        return 'hello world'
        
help(MyClass)

x = MyClass()
x
x.__class__
x.i
x.i = 0
x.i
x.f()



y = x.__class__
type(y)
y.__name__



class Complex:
    def __init__(self, realpart, imagpart):
        self.r = realpart
        self.i = imagpart
        
a = Complex(3,4)
a
a.r
a.i = 7
a.i
help(a)
a
b = Complex(7,7)
b.i

# --------------------------------------------------------------------------------

android_fcg_path = os.path.join("..", "..", "datasets", ("ANDROID FCG (2 "
                                "classes, 26 directed graphs, "
                                "unlabeled edges)"))
                                
fcg_clean_path = os.path.join(android_fcg_path, "clean")
fcg_mal_path = os.path.join(android_fcg_path, "malware")

os.listdir(fcg_clean_path)

