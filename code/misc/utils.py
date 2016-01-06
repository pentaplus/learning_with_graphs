import os
import sys
from itertools import tee

def has_elem(it): 
    it, any_check = tee(it)
    try:
        any_check.next()
        return True, it
    except StopIteration:
        return False, iter


def makedir(path):
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def clear_dicts_of_dict(d):
    for k in d.iterkeys():
        d[k].clear()
    
            
def write(string, result_file):
    sys.stdout.write(string)
    if result_file:
        result_file.write(string)
        
    

        


#    for k in d.iterkeys():
#        d[k] = {}
#    return