import inspect
import sys

import numpy as np
from os.path import abspath, dirname, join


# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from misc import utils


    

def extract_features(graph_meta_data_of_num, param_range = [None]):
    extr_start_time = time.time()
    
    data_mat_of_param = {}
    extr_time_of_param = {}    
    
    

        
    # initialize kernel_mat
    graphs_count = len(graph_meta_data_of_num)
    
    
    
    # iterate over all graphs in the dataset -------------------------------------
    for i, (graph_num, (G, class_lbl)) in\
                                    enumerate(graph_meta_data_of_num.iteritems()):
        pass
    
    data_mat_of_param[graphlet_size] = data_matrix
    
    extr_end_time = time.time()
    extr_time_of_param[graphlet_size] = extr_end_time - extr_start_time

    return data_mat_of_param, extr_time_of_param


if __name__ == '__main__':
    import time
    from misc import dataset_loader
    
    DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', 'datasets')
    dataset = 'MUTAG'
#    dataset = 'DD'
#    dataset = 'ENZYMES'
#    dataset = 'NCI1'
#    dataset = 'NCI109'
    graph_meta_data_of_num, class_lbls =\
                               dataset_loader.load_dataset(DATASETS_PATH, dataset)
    
    
    start = time.time()
    data_mat_of_param, extr_time_of_param =\
                                    extract_features(graph_meta_data_of_num, None)
    end = time.time()
    print end - start
    

