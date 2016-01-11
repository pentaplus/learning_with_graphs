import inspect
import networkx as nx
import numpy as np
import sys
import time

from numpy import array, bitwise_xor, float64, frombuffer, uint64
from os import urandom
from os.path import abspath, dirname, join
from scipy.sparse import csr_matrix


# determine script path
filename = inspect.getframeinfo(inspect.currentframe()).filename
script_path = dirname(abspath(filename))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(script_path, '..'))

from misc import datasetloader, utils


# test section -------------------------------------------------------------------

import timeit

setup1 = """
from os import urandom
from numpy import bitwise_xor, frombuffer, uint64
l = []
for i in xrange(1000):
    l.append(frombuffer(urandom(8), dtype = uint64))
"""

code1 = """
s = l[0]
for i in xrange(1, 1000):
    s = bitwise_xor(s, l[i])
"""

setup2 = """
from random import randint
l = []
for i in xrange(1000):
    l.append(randint(0, 18446744073709551615))
"""

code2 = """
s = l[0]
for i in xrange(1, 1000):
    s = s ^ i
"""




N = 1
min(timeit.repeat(code1, setup1, repeat = 3, number = N))/N
min(timeit.repeat(code2, setup2, repeat = 3, number = N))/N


np.frombuffer(urandom(8), dtype = uint64)

l = []
for i in xrange(1000):
    l.append(np.frombuffer(urandom(8), dtype = uint64))

s = l[0]
for i in xrange(1, 82):
    s = bitwise_xor(s, l[i])
    

for x in xrange(1000):
    x = np.frombuffer(urandom(8), dtype = uint64)
    
# http://www.falatic.com/index.php/108/python-and-bitwise-rotation
    
# max bits > 0 == width of the value in bits (e.g., int_16 -> 16)
 
# Rotate left: 0b1001 --> 0b0011
rol = lambda val, r_bits, max_bits: \
    (val << r_bits%max_bits) & (2**max_bits-1) | \
    ((val & (2**max_bits-1)) >> (max_bits-(r_bits%max_bits)))
 
# Rotate right: 0b1001 --> 0b1100
ror = lambda val, r_bits, max_bits: \
    ((val & (2**max_bits-1)) >> r_bits%max_bits) | \
    (val << (max_bits-(r_bits%max_bits)) & (2**max_bits-1))
 
max_bits = 16  # For fun, try 2, 17 or other arbitrary (positive!) values
 
print()
for i in xrange(0, max_bits*2-1):
    value = 0xC000
    newval = rol(value, i, max_bits)
    print("0x%08x << 0x%02x --> 0x%08x" % (value, i, newval))
 
print()
for i in xrange(0, max_bits*2-1):
    value = 0x0003
    newval = ror(value, i, max_bits)
    print("0x%08x >> 0x%02x --> 0x%08x" % (value, i, newval))

#---------------------------------------------------------------------------------

DATASETS_PATH = join(script_path, '..', '..', 'datasets')
dataset = 'MUTAG'
graph_of_num = datasetloader.load_dataset(DATASETS_PATH, dataset)

del filename
del script_path
del dataset

def extract_features(graph_of_num, h):
    # the keys are graph numbers and the values are lists of features
    features_dict = {}
    
    # the keys are graph numbers and the values are lists which contain the number
    # of occurences of the features corresponding to the feature at the same index
    # in the feature list in features_dict, that is
    # feature_counts_dict[graph_number][i] == number of occurences of feature
    # features_dict[graph_number][i]
    feature_counts_dict = {}
    
    # the keys are graph numbers and the values are dictionaries which map
    # features to their position in features_dict[graph_number] and
    # feature_counts_dict[graph_number], respectively
    index_of_lbl_dict = {}
    
    # the keys are graph numbers and the values are dictionaries which map
    # nodes to their updated label
    next_upd_lbls_dict = {}
    upd_lbls_dict = {}
    
    # keys are the node labels which are stored in the dataset and the values are
    # new compressed labels
    compr_func = {}
    
    # next_compr_lbl is used for assigning new compressed labels to the nodes
    # These build the features (= columns in data_matrix) used for the explicit
    # graph embedding
    next_compr_lbl = 0
    
    # initialize feature_counts_dict, features_dict, index_of_lbl_dict,
    # next_upd_lbls_dict and upd_lbls_dict
    for graph_num in graph_of_num.iterkeys():
        feature_counts_dict[graph_num] = []
        features_dict[graph_num] = []
        index_of_lbl_dict[graph_num] = {}
        next_upd_lbls_dict[graph_num] = {}
        upd_lbls_dict[graph_num] = {}
    
    
    
    # iterate over all graphs in the dataset -------------------------------------
    # !!
#    num_of_it = 5
    for r in xrange(h + 1):
        for (graph_num, (G, class_lbl)) in graph_of_num.iteritems():
            for v in G.nodes_iter():
                if r == 0:
                    uncompr_lbl = G.node[v]['label']
                else:
                    # r > 0
                    has_elem, neigh_iter = utils.has_elem(G.neighbors_iter(v))
                    if not has_elem:
                        # node v has no neighbors
                        continue
            
                    # determine the list of labels of the nodes adjacent to v
                    neigh_lbls = []
                    for v_neigh in neigh_iter:
                        neigh_lbls.append(upd_lbls_dict[graph_num][v_neigh])
                
                    # sort neigh_lbls in ascending order
                    if len(neigh_lbls) > 1:
                        neigh_lbls.sort()
                
                    # concatenate the neighboring labels to the label of v
                    uncompr_lbl = str(upd_lbls_dict[graph_num][v])
                    if len(neigh_lbls) == 1:
                        uncompr_lbl += ',' + str(neigh_lbls[0])
                    elif len(neigh_lbls) > 1:
                        uncompr_lbl += ',' + ','.join(map(str, neigh_lbls))
                        
                
                if not uncompr_lbl in compr_func:
                    # assign a compressed label new_compr_lbl to uncompr_lbl
                    new_compr_lbl = next_compr_lbl
                    compr_func[uncompr_lbl] = new_compr_lbl
                    next_compr_lbl += 1
                else:
                    # determine compressed label new_compr_lbl assigned to
                    # uncompr_lbl
                    new_compr_lbl = compr_func[uncompr_lbl]
        
                if new_compr_lbl not in index_of_lbl_dict[graph_num]:
                    # len(feature_counts_dict[graph_num])
                    # == len(features_dict[graph_num])
                    index = len(feature_counts_dict[graph_num])
        
                    index_of_lbl_dict[graph_num][new_compr_lbl] = index
        
                    # features_dict[graph_num][index]
                    # == feature upd_lbls_dict[graph_num][v] (== new_compr_lbl)
                    features_dict[graph_num].append(new_compr_lbl)
        
                    # set number of occurrences of the feature
                    # upd_lbls_dict[graph_num][v] (== new_compr_lbl) to 1
                    feature_counts_dict[graph_num].append(1)
                else:
                    # features_dict[graph_num][index]
                    # == feature upd_lbls_dict[graph_num][v] (== new_compr_lbl)
                    index = index_of_lbl_dict[graph_num][new_compr_lbl]
        
                    # increase number of occurrences of the feature
                    # upd_lbls_dict[graph_num][v] (== new_compr_lbl)
                    feature_counts_dict[graph_num][index] += 1
        
                # upd_lbls_dict[graph_num][v] == compr_func[lbl]
                # == new_compr_lbl
                next_upd_lbls_dict[graph_num][v] = new_compr_lbl
        
        if r > 0:
            # prepare upd_lbls_dict for reuse
            utils.clear_dicts_of_dict(upd_lbls_dict)
        dict_of_cleared_dicts = upd_lbls_dict
           
        upd_lbls_dict = next_upd_lbls_dict
        next_upd_lbls_dict = dict_of_cleared_dicts
    
   
   
## iterate over all graphs in the dataset !!! ---------------------------------
## r == 1
##for (graph_num, (G, class_lbl)) in graph_of_num.iteritems():
#graph_num = 1
#G = graph_of_num[1][0]
#for v in G.nodes_iter():
#    has_elements, nodes_iter = utils.has_elements(G.neighbors_iter(v))
#    if not has_elements:
#        # node v has no neighbors
#        continue
#
#    # determine the list of labels belonging to nodes adjacent to v
#    neigh_lbls = []
#    for v_neigh in nodes_iter:
#        neigh_lbls.append(upd_lbls_dict[graph_num][v_neigh])
#
#    # sort neigh_lbl_lst in ascending order
#    if len(neigh_lbls) > 1:
#        neigh_lbls.sort()
#
#    # concatenate the neighboring lbls to the lbl of v
#    s = str(upd_lbls_dict[graph_num][v])
#    if len(neigh_lbls) == 1:
#        s += ',' + str(neigh_lbls[0])
#    elif len(neigh_lbls) > 1:
#        s += ',' + ','.join(map(str, neigh_lbls))
#
#    # !! incorrect
#    if not s in compr_func:
#        new_compr_lbl = next_compr_lbl
#        compr_func[s] = next_compr_lbl        
#        next_compr_lbl += 1
#    else:
#        new_compr_lbl = compr_func[s]
#        
#    if new_compr_lbl not in index_of_lbl_dict[graph_num]:
#            # len(feature_counts_dict[graph_num])
#            # == len(features_dict[graph_num])
#            index = len(feature_counts_dict[graph_num])
#
#            index_of_lbl_dict[graph_num][new_compr_lbl] = index
#
#            # features_dict[graph_num][index]
#            # == feature upd_lbls_dict[graph_num][v] (== new_compr_lbl)
#            features_dict[graph_num].append(new_compr_lbl)
#
#            # set number of occurrences of the feature
#            # upd_lbls_dict[graph_num][v] (== new_compr_lbl) to 1
#            feature_counts_dict[graph_num].append(1)
#        else:
#            # features_dict[graph_num][index]
#            # == feature upd_lbls_dict[graph_num][v] (== new_compr_lbl)
#            index = index_of_lbl_dict[graph_num][new_compr_lbl]
#
#            # increase num of occurrences of the feature
#            # upd_lbls_dict[graph_num][v] (== new_compr_lbl)
#            feature_counts_dict[graph_num][index] += 1
#
#        # upd_lbls_dict[graph_num][v] == compr_func[lbl]
#        # == new_compr_lbl
#        upd_lbls_dict[graph_num][v] = new_compr_lbl
    
        


    # list containing the features of all graphs
    features = []
    
    # list containing the corresponding features counts of all graphs
    feature_counts = []
    
    # list indicating to which graph (= row in data_matrix) the features in
    # the list features belong. The difference feature_ptr[i+1] - feature_ptr[i]
    # equals the number of specified entries for row i. Consequently, the number
    # of rows of data_matrix equals len(feature_ptr) - 1.
    feature_ptr = [0]
    
    # list containing the class labels of all graphs
    class_lbls = []
    
    # !!
    del graph_num
    del class_lbl
    del v
    del uncompr_lbl
    del index
    del new_compr_lbl
    del next_compr_lbl
    
    
    for (graph_num, (G, class_lbl)) in graph_of_num.iteritems():
        features += features_dict[graph_num]
        feature_counts += feature_counts_dict[graph_num]
        feature_ptr.append(feature_ptr[-1] + len(features_dict[graph_num]))
    
        class_lbls.append(class_lbl)
    
    class_lbls = array(class_lbls)
    
    
    # data_matrix is of type csr_matrix and has the following form:
    # [feature vector of the first graph,
    #  feature vector of the second graph,
    #                .
    #                .
    #  feature vector of the last graph]
    data_matrix = csr_matrix((array(feature_counts), array(features),
                              array(feature_ptr)),
                              shape = (len(graph_of_num), len(compr_func)),
                              dtype = float64)
    
    # !! DEBUG
#    Z = data_matrix.todense()
    
    return data_matrix, class_lbls





#DATASETS_PATH = join(script_path, '..', '..', 'datasets')
#dataset = 'MUTAG'
#graph_of_num = datasetloader.load_dataset(DATASETS_PATH, dataset)
#data_matrix, class_lbls = extract_features(graph_of_num)





