"""
Android FCG subset picker.

This module provides functionality for choosing a subset of graphs
from the whole Android FCG dataset, which consists of 135,791 benign
and 12,158 malicious samples.
"""

__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__date__ = "2016-02-28"


import inspect
import os
import re
import shutil
import sys

from os.path import abspath, dirname, join
from random import shuffle


# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from misc import dataset_loader, utils

# ratio of benign samples within the subset
#RATIO = 135791./(135791 + 12158) # 0.91782
RATIO = 0.75
SUBSET_SIZE = 5000
# number of benign samples
CLASS_0_SUBSET_SIZE = int(RATIO * SUBSET_SIZE)
# number of malicious samples
CLASS_1_SUBSET_SIZE = SUBSET_SIZE - CLASS_0_SUBSET_SIZE

SOURCE_CLASSES_PATH = 'Z:\ANDROID FCG\pz'
                           
utils.check_for_pz_folder()
                           
os.makedirs('pz')

folder_of_class = dataset_loader.get_folder_of_class_dict(SOURCE_CLASSES_PATH)
                

for class_lbl, class_folder in folder_of_class.iteritems():
    source_class_path = join(SOURCE_CLASSES_PATH, class_folder)
    target_class_path = join('pz', class_folder)
    os.makedirs(target_class_path)
    
    graph_file_names = utils.list_files(source_class_path)
    shuffle(graph_file_names)
    
    if class_lbl == 0:
        graph_file_names_subset = graph_file_names[:CLASS_0_SUBSET_SIZE]
    
    if class_lbl == 1:
        graph_file_names_subset = graph_file_names[:CLASS_1_SUBSET_SIZE]
    
    copied_graph_files_count = 0
    # copy graph files of the chosen subset to destination folder
    for graph_file_name in graph_file_names_subset:
        graph_file_base_name = re.match('.*?(?=\.)', graph_file_name).group(0)
        
        shutil.copyfile(join(source_class_path, graph_file_name),
                        join(target_class_path, graph_file_base_name + '.pz'))
                        
        copied_graph_files_count += 1
        
        if copied_graph_files_count % 10 == 0:
            print "Graph files copied: %d" % copied_graph_files_count
            

        
