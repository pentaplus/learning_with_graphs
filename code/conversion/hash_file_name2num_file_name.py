import inspect
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


DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', 'datasets')

ANDROID_FCG_PARTIAL_PATH = join(DATASETS_PATH, ('ANDROID FCG PARTIAL (2 '
                                'classes, x directed graphs, unlabeled edges)'),
                                'pz')
                                
# !! pz missing
CFG_PATH = join(DATASETS_PATH, ('CFG (2 classes, x directed graphs, unlabeled '
                                'edges)'))
                                

DATASET_PATH = ANDROID_FCG_PARTIAL_PATH
#DATASET_PATH = CFG_PATH

class_folders = os.listdir(DATASET_PATH)

graph_num = 0
with open(join(DATASET_PATH, 'hash_num_map.txt'), 'w') as f:
    for class_folder in class_folders:
        graph_files_path = join(DATASET_PATH, class_folder)
        graph_file_names = os.listdir(graph_files_path)
        
        for graph_file_name in graph_file_names:
            os.rename(join(graph_files_path, graph_file_name),
                      join(graph_files_path, str(graph_num) + '.pz'))
                      
            hash_part = re.search('.*?(?=\.)', graph_file_name).group(0)
            f.write(hash_part + ': ' + str(graph_num) + '\n')          
            
            graph_num += 1


    