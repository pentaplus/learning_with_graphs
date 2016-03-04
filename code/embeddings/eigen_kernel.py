"""
Weisfeiler-Lehman subtree kernel.

This module provides the function extract_features for the
corresponding feature extraction.
"""

__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__date__ = "2016-03-04"


import inspect
import numpy as np
import sys
import time

from numpy.linalg import eigvalsh
from os.path import abspath, dirname, join


# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from misc import utils, pz


def get_average_num_of_nodes(graph_meta_data_of_num):
    node_counts = []
    for graph_path, class_lbl in graph_meta_data_of_num.itervalues():
        G = pz.load(graph_path)
        node_counts.append(G.number_of_nodes())
        
    return np.mean(node_counts)
    

def extract_features(graph_meta_data_of_num, param_range = [None]):
    extr_start_time = time.time()
    
    feature_mat_of_param = {}
    extr_time_of_param = {}
    
    num_graphs = len(graph_meta_data_of_num)
    
    average_num_of_nodes = get_average_num_of_nodes(graph_meta_data_of_num)

    feature_mat = np.zeros((num_graphs, int(average_num_of_nodes)),
                           dtype = np.float64)
    
    #=============================================================================
    # 1) extract features iterating over all graphs in the dataset
    #=============================================================================
    for i, (graph_path, class_lbl) in \
            enumerate(graph_meta_data_of_num.itervalues()):
                
        # !!
#        if i % 10 == 0:
#            print i
        
        # load graph
        G = pz.load(graph_path)
        # determine its adjacency matrix
        A = utils.get_adjacency_matrix(G)
#        A = nx.adj_matrix(G, weight = None)
        
        # !!
        import sys
        sys.modules['__main__'].A = A
        sys.modules['__main__'].F = feature_mat
        
        x = 0
        eigvalsh(A)

        for j in xrange(feature_mat.shape[1]):
            largest_eigen_val = eigvalsh(A)[-1]
        
        
        # feature_mat is of type csr_matrix and has the following form:
        # [feature vector of the first graph,
        #  feature vector of the second graph,
        #                .
        #                .
        #  feature vector of the last graph]
#        feature_mat = csr_matrix((np.array(feature_counts), np.array(features),
#                                  np.array(feature_ptr)),
#                                  shape = (len(graph_meta_data_of_num),
#                                  len(compr_func)), dtype = np.float64)
        
        feature_mat_of_param[None] = feature_mat
        
        extr_end_time = time.time()
        extr_time = extr_end_time - extr_start_time
        
        extr_time_of_param[None] = extr_time
  
   
    return feature_mat_of_param, extr_time_of_param



# !!
if __name__ == '__main__':
    from misc import dataset_loader as loader
    
    DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', 'datasets')
    dataset = 'MUTAG'
#    dataset = 'PTC(MR)'
    
    graph_meta_data_of_num, class_lbls =\
        loader.get_graph_meta_data_of_num_dict_and_class_lbls(dataset,
                                                              DATASETS_PATH)    
    
    feature_mat_of_param, extr_time_of_param \
        = extract_features(graph_meta_data_of_num, [None])
                                 
    feature_mat = feature_mat_of_param[None]                                                                
                                                                   

    
    

