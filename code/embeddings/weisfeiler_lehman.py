import inspect
import numpy as np
import sys
import time

from collections import defaultdict
from os.path import abspath, dirname, join
from scipy.sparse import csr_matrix


# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from misc import utils


def extract_features(graph_of_num, h_range):
    extraction_start_time = time.time()
    
    data_mat_of_param = {}
    extraction_time_of_param = {}
    mat_construction_times = []
    
    h_max = max(h_range)
    
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
    # new compressed labels
    compr_func = {}
    
    # next_compr_lbl is used for assigning new compressed labels to the nodes
    # These build the features (= columns in data_mat) used for the explicit
    # graph embedding
    next_compr_lbl = 0
    
    
    # ----------------------------------------------------------------------------
    # 1) extract features iterating over all graphs in the dataset
    # ----------------------------------------------------------------------------
    for r in h_range:
        for (graph_num, (G, class_lbl)) in graph_of_num.iteritems():
            # !!        
#            if graph_num % 100 == 0:
#                print 'r = ' + str(r) + ', graph_num = ' + str(graph_num)
                
            for v in G.nodes_iter():
                if r == 0:
                    uncompr_lbl = G.node[v]['label']
                    if isinstance(uncompr_lbl, np.ndarray):
                        uncompr_lbl = utils.calc_hash_of_array(uncompr_lbl)
                else:
                    # r > 0
                    has_elem, nbrs_iter = utils.has_elem(G.neighbors_iter(v))
                    if not has_elem:
                        # node v has no neighbors
                        next_upd_lbls_dict[graph_num][v] =\
                                                       upd_lbls_dict[graph_num][v]
                        continue
            
                    # determine the list of labels of the nodes adjacent to v
                    nbrs_lbls = []
                    for v_nbr in nbrs_iter:                            
                        nbrs_lbls.append(upd_lbls_dict[graph_num][v_nbr])
                
                    # sort nbrs_lbls in ascending order
                    if len(nbrs_lbls) > 1:
                        nbrs_lbls.sort()
                
                    # concatenate the neighboring labels to the label of v
                    uncompr_lbl = str(upd_lbls_dict[graph_num][v])
                    if len(nbrs_lbls) == 1:
                        uncompr_lbl += ',' + str(nbrs_lbls[0])
                    elif len(nbrs_lbls) > 1:
                        uncompr_lbl += ',' + ','.join(map(str, nbrs_lbls))
                        
                
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
                
                if r < h_max:
                    # next_upd_lbls_dict[graph_num][v] == compr_func[lbl]
                    # == new_compr_lbl
                    next_upd_lbls_dict[graph_num][v] = new_compr_lbl
        
        # ------------------------------------------------------------------------
        # 2) construct data matrix whose i-th row equals the i-th feature vector,
        #    which comprises the features of the first r iterations
        # ------------------------------------------------------------------------
        mat_construction_start_time = time.time()        
        
        # list containing the features of all graphs
        features = []
        
        # list containing the corresponding features counts of all graphs
        feature_counts = []
        
        # list indicating to which graph (= row in data_mat) the features in
        # the list features belong. The difference
        # feature_ptr[i+1] - feature_ptr[i] equals the number of specified entries
        # for row i. Consequently, the number of rows of data_mat equals
        # len(feature_ptr) - 1.
        feature_ptr = [0]
        
        
        for graph_num in graph_of_num.iterkeys():
            features += features_dict[graph_num]
            feature_counts += feature_counts_dict[graph_num]
            feature_ptr.append(feature_ptr[-1] + len(features_dict[graph_num]))
        
        
        # data_mat is of type csr_matrix and has the following form:
        # [feature vector of the first graph,
        #  feature vector of the second graph,
        #                .
        #                .
        #  feature vector of the last graph]
        data_mat = csr_matrix((np.array(feature_counts), np.array(features),
                               np.array(feature_ptr)), shape = (len(graph_of_num),
                               len(compr_func)), dtype = np.float64)
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
    
   
    return data_mat_of_param, extraction_time_of_param



# !!
if __name__ == '__main__':
    from misc import dataset_loader
    
    DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', 'datasets')
    dataset = 'MUTAG'
    graph_of_num, class_lbls = dataset_loader.load_dataset(DATASETS_PATH, dataset)    
    
    del SCRIPT_PATH
    del SCRIPT_FOLDER_PATH
    del dataset
    
    h_range = range(6)
    
    start = time.time()
    data_mat_of_param, extraction_time_of_param = extract_features(graph_of_num,
                                                                   h_range)
    end = time.time()
    print end - start
    
    
    from sklearn.svm import SVC
    from sklearn.cross_validation import cross_val_score
    from sklearn.metrics.pairwise import pairwise_kernels
    
    #X = []
    #for i in xrange(len(graph_of_num)):
    #    X.append([i])
    #X = np.array(X)
    #    
    #
    #data = ['aab', 'aaabb']
    #    
    #def my_kernel(X, Y):
    #    '''This function is used to pre-compute the kernel matrix from data matrices;
    #       that matrix should be an array of shape (n_samples, n_samples).'''
    ##    print 'X', X
    ##    print 'X type', type(X)
    ##    print 'X size', X.shape
    ##    print 'Y', Y
    ##    print 'Y type', type(Y)
    ##    print 'Y size', Y.shape
    #    i = int(X[0,0])
    #    j = int(Y[1,0])
    ##    return data[i].count('a')*data[j].count('a') +\
    ##           data[i].count('b')*data[j].count('b')
    #    return np.array([[1, 2], [2,3]])
               
        
    #clf = SVC(kernel = 'precomputed')
    #
    #clf.fit(pairwise_kernels(data_mat), class_lbls)
    #
    #cross_val_score(clf, X, class_lbls, cv = 10)
    
    #for i, (num, tup) in enumerate(graph_of_num.iteritems()):
    #    print i, num, tup