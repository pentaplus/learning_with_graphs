import inspect
import itertools as itools
import numpy as np
import sys
import time

from collections import defaultdict
from os.path import abspath, dirname, join
from random import randint
from scipy.sparse import csr_matrix


# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from misc import utils


# test section -------------------------------------------------------------------

#import timeit
#
#setup1 = """
#from os import urandom
#from numpy import bitwise_xor, frombuffer, uint64
#l = []
#for i in xrange(1000):
#    l.append(frombuffer(urandom(8), dtype = uint64))
#"""
#
#code1 = """
#s = l[0]
#for i in xrange(1, 1000):
#    s = bitwise_xor(s, l[i])
#"""
#
#setup2 = """
#from random import randint
#l = []
#for i in xrange(1000):
#    l.append(randint(0, 18446744073709551615))
#"""
#
#code2 = """
#s = l[0]
#for i in xrange(1, 1000):
#    s = s ^ i
#"""
#
#2**64-1 - 18446744073709551615
#%timeit randint(0, 18446744073709551615)
#
#%timeit frombuffer(urandom(8), dtype = uint64)
#
#N = 1
#min(timeit.repeat(code1, setup1, repeat = 3, number = N))/N
#min(timeit.repeat(code2, setup2, repeat = 3, number = N))/N
#
#
#np.frombuffer(urandom(8), dtype = uint64)
#
#l = []
#for i in xrange(1000):
#    l.append(np.frombuffer(urandom(8), dtype = uint64))
#
#s = l[0]
#for i in xrange(1, 82):
#    s = bitwise_xor(s, l[i])
#    
#
#for x in xrange(1000):
#    x = np.frombuffer(urandom(8), dtype = uint64)
    
# http://www.falatic.com/index.php/108/python-and-bitwise-rotation
    
# max bits > 0 == width of the value in bits (e.g., int_16 -> 16)
 
# Rotate left: 0b1001 --> 0b0011
#rol = lambda val, r_bits, max_bits: \
#    (val << r_bits % max_bits) & (2**max_bits - 1) | \
#    ((val & (2**max_bits - 1)) >> (max_bits - (r_bits % max_bits)))
# 
## Rotate right: 0b1001 --> 0b1100
#ror = lambda val, r_bits, max_bits: \
#    ((val & (2**max_bits - 1)) >> r_bits % max_bits) | \
#    (val << (max_bits - (r_bits%max_bits)) & (2**max_bits - 1))
# 
#max_bits = 64  # For fun, try 2, 17 or other arbitrary (positive!) values
# 
#print()
#for i in xrange(0, 16):
#    value = 0xC000
#    newval = rol(value, i, max_bits)
#    print "{0:016b} {1:016b} {2:016b}".format(value, i, newval)
# 
#print()
#for i in xrange(0, 16):
#    value = 0x0003
#    newval = ror(value, i, max_bits)
#    print "{0:064b} {1:04b} {2:064b}".format(value, i, newval)

#---------------------------------------------------------------------------------



def extract_features(graph_of_num, h_range, count_sensitive = True,
                     all_iter = False):
    extraction_start_time = time.time()
    
    data_mat_of_param = {}
    extraction_time_of_param = {}
    mat_construction_times = []
    
    h_max = max(h_range)                         
                        
    BIT_LBL_LEN = 16
    
    # rotate left
    rot_left = lambda val, r_bits: \
        (val << r_bits % BIT_LBL_LEN) & (2**BIT_LBL_LEN - 1) | \
        ((val & (2**BIT_LBL_LEN - 1)) >> (BIT_LBL_LEN - (r_bits % BIT_LBL_LEN)))
    
    # the keys are graph numbers and the values are lists of features
    features_dict = defaultdict(list)
    
    # the keys are graph numbers and the values are lists which contain the number
    # of occurences of the features corresponding to the feature at the same index
    # in the feature list in features_dict, that is
    # feature_counts_dict[graph_number][i] == number of occurences of feature
    # features_dict[graph_number][i]
    feature_counts_dict = defaultdict(list)
    
    # the keys are graph numbers and the values are dictionaries which map
    # features to their position in features_dict[graph_number] and
    # feature_counts_dict[graph_number], respectively
    index_of_lbl_dict = defaultdict(dict)
    
    # the keys are graph numbers and the values are dictionaries which map
    # nodes to their updated label
    next_upd_lbls_dict = defaultdict(dict)
    upd_lbls_dict = defaultdict(dict)
    
    # keys are the node labels which are stored in the dataset and the values are
    # 64-bit integers
    label_map = {}
    
    
    # ----------------------------------------------------------------------------
    # 1) extract features iterating over all graphs in the dataset
    # ----------------------------------------------------------------------------
    for r in h_range:
        for (graph_num, (G, class_lbl)) in graph_of_num.iteritems():
            for v in G.nodes_iter():
                if r == 0:
                    orig_lbl = G.node[v]['label']
                    
                    if isinstance(orig_lbl, np.ndarray):
                        orig_lbl = utils.calc_hash_of_array(orig_lbl)
                        
                    if not orig_lbl in label_map.iterkeys():
                        # assign a random bit label new_bit_lbl to orig_lbl
                        new_bit_lbl = randint(1, 2**BIT_LBL_LEN - 1)
                        label_map[orig_lbl] = new_bit_lbl
                    else:
                        # determine bit label new_bit_lbl assigned to orig_lbl
                        new_bit_lbl = label_map[orig_lbl]
                else:
                    # r > 0
                    has_elem, nbrs_iter = utils.has_elem(G.neighbors_iter(v))
                    if not has_elem:
                        # node v has no neighbors
                        next_upd_lbls_dict[graph_num][v] =\
                                                       upd_lbls_dict[graph_num][v]
                        continue
                    
                    if not count_sensitive:
                        # apply simple neighborhood hash
                        new_bit_lbl = rot_left(upd_lbls_dict[graph_num][v], 1)
                        for v_nbr in nbrs_iter:
                            new_bit_lbl ^= upd_lbls_dict[graph_num][v_nbr]
                    else:
                        # determine the list of labels of the nodes adjacent to v
                        nbrs_lbls = []
                        for v_nbr in nbrs_iter:
                            nbrs_lbls.append(upd_lbls_dict[graph_num][v_nbr])
                            
                        # determine the number of occurences of each neighbor
                        # label
                        num_of_nbr_lbl = {}
                        if len(nbrs_lbls) == 1:
                            nbr_lbl = nbrs_lbls[0]
                            num_of_nbr_lbl[nbr_lbl] = 1                  
                        else:
                            # len(nbrs_lbls) > 1
                            # sort nbrs_lbls in ascending order
                            nbrs_lbls.sort()
                            
                            prev_nbr_lbl = nbrs_lbls[0]
                            c = 1
                            for nbr_lbl in nbrs_lbls[1:]:
                                if nbr_lbl == prev_nbr_lbl:
                                    c += 1
                                else:
                                    num_of_nbr_lbl[prev_nbr_lbl] = c
                                    prev_nbr_lbl = nbr_lbl
                                    c = 1
                            num_of_nbr_lbl[nbr_lbl] = c
  
                        
                        # apply count sensitive neighborhood hash
                        new_bit_lbl = rot_left(upd_lbls_dict[graph_num][v], 1)
                        for nbr_lbl, num in num_of_nbr_lbl.iteritems():
                            new_bit_lbl ^= rot_left(nbr_lbl ^ num, num)
                
                if r < h_max:
                    # next_upd_lbls_dict[graph_num][v] == label_map[lbl]
                    # == new_bit_lbl
                    next_upd_lbls_dict[graph_num][v] = new_bit_lbl
                
                if new_bit_lbl not in index_of_lbl_dict[graph_num]:
                    # len(feature_counts_dict[graph_num])
                    # == len(features_dict[graph_num])
                    index = len(feature_counts_dict[graph_num])
        
                    index_of_lbl_dict[graph_num][new_bit_lbl] = index
        
                    # features_dict[graph_num][index]
                    # == feature upd_lbls_dict[graph_num][v] (== new_bit_lbl)
                    features_dict[graph_num].append(new_bit_lbl)
        
                    # set number of occurrences of the feature
                    # upd_lbls_dict[graph_num][v] (== new_bit_lbl) to 1
                    feature_counts_dict[graph_num].append(1)
                else:
                    # features_dict[graph_num][index]
                    # == feature upd_lbls_dict[graph_num][v] (== new_bit_lbl)
                    index = index_of_lbl_dict[graph_num][new_bit_lbl]
        
                    # increase number of occurrences of the feature
                    # upd_lbls_dict[graph_num][v] (== new_bit_lbl)
                    feature_counts_dict[graph_num][index] += 1
                    
        
        # ------------------------------------------------------------------------
        # 2) compress bit labels and construct data matrix whose i-th row equals
        #    the i-th feature vector
        # ------------------------------------------------------------------------
        mat_construction_start_time = time.time()
        
        # list containing the features of all graphs
        features = []
		
        # list containing the corresponding features counts of all graphs
        feature_counts = []
		
        # list indicating to which graph (= row in data_mat) the features in the
        # list features belong. The difference feature_ptr[i+1] - feature_ptr[i]
        # equals the number of specified entries for row i. Consequently, the
        # number of rows of data_mat equals len(feature_ptr) - 1.
        feature_ptr = [0]
		
        # keys are the bit labels and the values are new compressed labels
        compr_func = {}
		
        # next_compr_lbl is used for assigning new compressed labels to the nodes.
        # These build the features (= columns in data_mat), which are used for the
        # explicit graph graph embedding.
        next_compr_lbl = 0
		
	
        for (graph_num, (G, class_lbl)) in graph_of_num.iteritems():
            for bit_lbl, bit_lbl_count in\
                                      itools.izip(features_dict[graph_num],
                                                  feature_counts_dict[graph_num]):
				  if not bit_lbl in compr_func:
					  compr_func[bit_lbl] = next_compr_lbl
					  compr_lbl = next_compr_lbl
					  next_compr_lbl += 1
				  else:
					  compr_lbl = compr_func[bit_lbl]
					
				  features.append(compr_lbl)
				  feature_counts.append(bit_lbl_count)
				
				
            feature_ptr.append(feature_ptr[-1] + len(features_dict[graph_num]))
		  
		  
        # data_mat is of type csr_matrix and has the following form:
        # [feature vector of the first graph,
        #  feature vector of the second graph,
        #                .
        #                .
        #  feature vector of the last graph]
        data_mat = csr_matrix((np.array(feature_counts), np.array(features),
                               np.array(feature_ptr)),
                               dtype = np.float64)
        data_mat_of_param[r] = data_mat
        
        
        extraction_end_time = time.time()
        extraction_time = extraction_end_time - extraction_start_time -\
                          sum(mat_construction_times)
        
        mat_construction_end_time = time.time()
        mat_construction_time =\
                           mat_construction_end_time - mat_construction_start_time
        mat_construction_times.append(mat_construction_time)
        
        extraction_time += mat_construction_time
        extraction_time_of_param[r] = extraction_time
		  
		  
        if r < h_max:
            upd_lbls_dict = next_upd_lbls_dict
            next_upd_lbls_dict = defaultdict(dict)
            
            if not all_iter:
                features_dict = defaultdict(list)
                feature_counts_dict = defaultdict(list)
                index_of_lbl_dict = defaultdict(dict)

    return data_mat_of_param, extraction_time_of_param
    

# !!
if __name__ == '__main__':
    from misc import dataset_loader
        
    DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', 'datasets')
    dataset = 'MUTAG'
    graph_of_num, class_lbls = dataset_loader.load_dataset(DATASETS_PATH, dataset)
    
    
    # h = 0: 56900    586
    # h = 1: 55892    828
    # h = 2: 63964    947
    # h = 3: 62689   1010
    # h = 4: 65162    929
    # h = 5: 64494    979
    # h = 6: 61520    964
    # h = 7: 63481   1009
    # h = 8: 63804    970
    # h = 9: 63322   1003
    # h =10: 62836    950
    
    h_range = range(6)
    start = time.time()
    data_mat_of_param, extraction_time_of_param =\
                   extract_features(graph_of_num, h_range, count_sensitive = True,
                                    all_iter = True)
    end = time.time()
    print 'h_range = %s: %.3f' % (h_range, end - start)
    
    
#    Z = data_mat.todense()
#    
#    print data_mat.__repr__()
#    #print data_mat.__str__()