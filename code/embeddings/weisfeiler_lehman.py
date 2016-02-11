import inspect
import numpy as np
import sys

from os.path import abspath, dirname, join
from scipy.sparse import csr_matrix


# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from misc import utils



def extract_features(graph_of_num, max_h):
    data_matrices = []
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
    for r in xrange(max_h + 1):
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
                
                if r < max_h:
                    # next_upd_lbls_dict[graph_num][v] == compr_func[lbl]
                    # == new_compr_lbl
                    next_upd_lbls_dict[graph_num][v] = new_compr_lbl
        
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
    #    del graph_num
    #    del class_lbl
    #    del v
    #    del uncompr_lbl
    #    del index
    #    del new_compr_lbl
    #    del next_compr_lbl
        
        
        for (graph_num, (G, class_lbl)) in graph_of_num.iteritems():
            features += features_dict[graph_num]
            feature_counts += feature_counts_dict[graph_num]
            feature_ptr.append(feature_ptr[-1] + len(features_dict[graph_num]))
        
            class_lbls.append(class_lbl)
        
        class_lbls = np.array(class_lbls)
        
        
        # data_matrix is of type csr_matrix and has the following form:
        # [feature vector of the first graph,
        #  feature vector of the second graph,
        #                .
        #                .
        #  feature vector of the last graph]
        data_matrix = csr_matrix((np.array(feature_counts), np.array(features),
                                  np.array(feature_ptr)),
                                  shape = (len(graph_of_num), len(compr_func)),
                                  dtype = np.float64)
        data_matrices.append(data_matrix)
    
    # !! DEBUG
#    Z = data_matrix.todense()
    
  
        if r < max_h:
            if r > 0:
                # prepare upd_lbls_dict for reuse
                utils.clear_dicts_of_dict(upd_lbls_dict)
            dict_of_cleared_dicts = upd_lbls_dict
               
            upd_lbls_dict = next_upd_lbls_dict
            next_upd_lbls_dict = dict_of_cleared_dicts
    
   
    return data_matrices, class_lbls      



# !!
if __name__ == '__main__':
    from misc import dataset_loader
    
    DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', 'datasets')
    dataset = 'MUTAG'
    graph_of_num = dataset_loader.load_dataset(DATASETS_PATH, dataset)    
    
    del SCRIPT_PATH
    del SCRIPT_FOLDER_PATH
    del dataset
    
    import time
    start = time.time()
    data_matrix, class_lbls = extract_features(graph_of_num, 10)
    end = time.time()
    print end-start
    
    
    import numpy as np
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
    #clf.fit(pairwise_kernels(data_matrix), class_lbls)
    #
    #cross_val_score(clf, X, class_lbls, cv = 10)
    
    #for i, (num, tup) in enumerate(graph_of_num.iteritems()):
    #    print i, num, tup