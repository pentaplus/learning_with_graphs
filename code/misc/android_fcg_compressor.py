import inspect
import networkx as nx
import numpy as np
import os
import re
import sys

from os.path import abspath, dirname, join



# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from misc import pz, utils

def bin_array_to_num(array):
    array_str = ''
    for i in xrange(array.shape[0]):
        array_str += str(array[i])
        
    return int(array_str, 2)


DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', 'datasets')

SOURCE_CLASSES_PATH = join(DATASETS_PATH, ('ANDROID FCG PARTIAL (2 '
                                'classes, x directed graphs, unlabeled edges)'),
                                'pz')

utils.check_for_pz_folder()
                           
os.makedirs('pz')

class_folders = utils.list_sub_dirs(SOURCE_CLASSES_PATH)

graph_num = 0
with open(join(SOURCE_CLASSES_PATH, 'hash_num_map.txt'), 'w') as f:
    for class_folder in class_folders:
        source_class_path = join(SOURCE_CLASSES_PATH, class_folder)
        target_class_path = join('pz', class_folder)
        os.makedirs(target_class_path)
        
        graph_file_names = utils.list_files(source_class_path)
        
        for graph_file_name in graph_file_names:
            id_to_num_mapper = utils.Id_to_num_mapper()
            G_uncompr = pz.load(join(source_class_path, graph_file_name))
            G_compr = nx.DiGraph()
            
            id_to_num_mapper = utils.Id_to_num_mapper()
            
            for node_id_tuple, lbl_array in G_uncompr.node.iteritems():
                node_id = '\n'.join(node_id_tuple)
                node_num = id_to_num_mapper.map_id_to_num(node_id)
                lbl_num = bin_array_to_num(lbl_array)
                
                G_compr[node_num] = lbl_num
            
                