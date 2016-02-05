import inspect
import networkx as nx
import numpy as np
import sys

from numpy import array, float64
from os.path import abspath, dirname, join
from scipy.sparse import csr_matrix


# determine script path
filename = inspect.getframeinfo(inspect.currentframe()).filename
script_path = dirname(abspath(filename))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(script_path, '..'))



def extract_features(graph_of_num):
    # the keys are graph numbers and the values are lists of features
    features_dict = {}

    # the keys are graph numbers and the values are lists which contain the number
    # of occurences of the features corresponding to the feature at the same index
    # in the feature list in features_dict, that is
    # feature_counts_dict[graph_num][i] == number of occurences of feature
    # features_dict[graph_num][i]
    feature_counts_dict = {}

    # the keys are graph numbers and the values are dictionaries which map
    # features to their position in features_dict[graph_num] and
    # feature_counts_dict[graph_num], respectively
    index_of_lbl_dict = {}

    # the keys are graph numbers and the values are lists containing the updated
    # labels
    upd_lbls_dict = {}

    # keys are the node labels which are stored in the dataset and the values are
    # new compressed labels
    compr_func = {}

    # next_compr_lbl is used for assigning new compressed labels to the nodes
    # These build the features (= columns in data_matrix) used for the explicit
    # graph embedding
    next_compr_lbl = 0

    # initialize feature_counts_dict, features_dict, index_of_lbl_dict and
    # upd_lbls_dict
    for graph_num in graph_of_num.iterkeys():
        feature_counts_dict[graph_num] = []
        features_dict[graph_num] = []
        index_of_lbl_dict[graph_num] = {}
        upd_lbls_dict[graph_num] = {}


    # iterate over all graphs in the dataset -------------------------------------
    # r == 0
    for (graph_num, (G, class_lbl)) in graph_of_num.iteritems():
        for v in G:
            uncompr_lbl = G.node[v]['label']
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

                # increase number of occurrences of the feature
                # upd_lbls_dict[graph_num][v] (== new_compr_lbl)
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
            upd_lbls_dict[graph_num][v] = new_compr_lbl


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
    Z = data_matrix.todense()

    return data_matrix, class_lbls
