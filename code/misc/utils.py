import os
import shutil
import sys
import time

from itertools import tee
from os import listdir
from os.path import isdir, isfile, join


def calc_hash_of_array(array):
    array.flags.writeable = False
    return hash(array.data)
    
    
def check_for_pz_folder():
    if isdir('pz'):
        shutil.rmtree('pz') # !!
        return        
        
        user_input = raw_input(('The directory \'pz\' already exists. '
                                'Do you want to delete it (y/n)? ')).strip()
        while True:
            if user_input == 'y':
                shutil.rmtree('pz')
                time.sleep(1)
                break
            if user_input == 'n':
                sys.exit(1)
            
            user_input = raw_input(('Invalid input! The directory \'pz\' already '
                                    'exists. Do you want to delete it '
                                    '(y/n)? ')).strip()
    
    
def clear_dicts_of_dict(d):
    for k in d.iterkeys():
        d[k].clear()
        

def fatal_error(msg, fid = None):
    print('Fatal error: ' + msg)
    
    if fid != None:
        fid.close()
    
    sys.exit(1)


def has_elem(it): 
    it, any_check = tee(it)
    try:
        any_check.next()
        return True, it
    except StopIteration:
        return False, iter

        
def list_files(path):
    return [f for f in listdir(path) if isfile(join(path, f))]


def makedir(path):
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

  
def write(string, result_file):
    sys.stdout.write(string)
    result_file.write(string)
        
    

        


#    for k in d.iterkeys():
#        d[k] = {}
#    return