# planed procedure:
# 1. test h = 1 in WEISFEILER_LEHMAN


import importlib
import inspect
import networkx as nx
import numpy as np
import time

from numpy import array, float64
from os.path import abspath, dirname, join
from scipy.sparse import csr_matrix
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# determine script path
FILE_NAME = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_PATH = dirname(abspath(FILE_NAME))

# !!
del FILE_NAME

from misc import datasetloader, utils
from performance_evaluation import perform_eval


def load_dataset(dataset, datasets_path):
    dataset_loading_start_time = time.time()

    graph_of_num = datasetloader.load_dataset(datasets_path, dataset)

    dataset_loading_end_time = time.time()
    dataset_loading_time = dataset_loading_end_time - dataset_loading_start_time
    print 'Loading the dataset %s took %.1f seconds.\n' % (dataset,
                                                           dataset_loading_time)
    return graph_of_num


def extract_features(graph_of_num, embedding, embedding_param, result_file):
    utils.write('-------------------------------------------------------------\n',
                result_file)
    utils.write('Parameter: %d \n\n' % embedding_param, result_file)
    feat_extr_start_time = time.time()

    data_matrix, class_lbls = embedding.extract_features(graph_of_num,
                                                         embedding_param)

    feat_extr_end_time = time.time()
    feat_extr_time = feat_extr_end_time - feat_extr_start_time
    utils.write('Feature extraction took %.1f seconds.\n' % feat_extr_time,
                result_file)
    print ''

    return data_matrix, class_lbls
    



start_time = time.time()

DATASETS_PATH = join(SCRIPT_PATH, '..', 'datasets')

WEISFEILER_LEHMAN = 'weisfeiler_lehman'
LABEL_COUNTER = 'label_counter'


#EMBEDDING_NAMES = [LABEL_COUNTER]
#EMBEDDING_NAMES = [WEISFEILER_LEHMAN, LABEL_COUNTER]
EMBEDDING_NAMES = [WEISFEILER_LEHMAN]

# keys are indices of the list EMBEDDING_NAMES, values are the respective
# parameters
#EMBEDDING_PARAMS = {WEISFEILER_LEHMAN : [1, 2, 3]}
#EMBEDDING_PARAMS = {WEISFEILER_LEHMAN : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
EMBEDDING_PARAMS = {WEISFEILER_LEHMAN : [0, 1, 2]}
#EMBEDDING_PARAMS = {WEISFEILER_LEHMAN : [2]}

#DATASET = 'ANDROID FCG' # !! change file names from hashes to numbers
#DATASET = 'CFG' # !! change file names from hashes to numbers

#DATASETS = ['DD', 'ENZYMES', 'MUTAG', 'NCI1', 'NCI109', 'PTC(MR)']
DATASETS = ['MUTAG']
#DATASETS = ['ENZYMES']
#DATASETS = ['MUTAG', 'PTC(MR)']

OPT_PARAM = True
#OPT_PARAM = False

#OPT = True
OPT = False

COMPARE_PARAM = True
#COMPARE_PARAM = False

#KERNELS = ['linear', 'rbf', 'poly', 'sigmoid']
#KERNELS = ['linear', 'rbf', 'sigmoid']
KERNELS = ['linear', 'rbf']

STRAT_KFOLD_VALUES = [False, True]
#STRAT_KFOLD_VALUES = [False]
#STRAT_KFOLD_VALUES = [True]

NUM_IT = 2

NUM_FOLDS = 10
#for embedding_name, dataset in itertools.product(EMBEDDING_NAMES, DATASETS):
for dataset in DATASETS:
    # ----------------------------------------------------------------------------
    # 1) load dataset
    # ----------------------------------------------------------------------------
    graph_of_num = load_dataset(dataset, DATASETS_PATH)
    
    
    for embedding_name in EMBEDDING_NAMES:
        result_path = join(SCRIPT_PATH, '..', 'results', embedding_name)
        utils.makedir(result_path)
        result_file = open(join(result_path, dataset + '.txt'), 'w')
        utils.write('NUMBER OF ITERATIONS: %d\n' % NUM_IT, result_file)
        print ''
        
        embedding = importlib.import_module('embeddings.' + embedding_name)
        
        if OPT_PARAM:
            param_range = EMBEDDING_PARAMS[embedding_name]
            
            for kernel in KERNELS:
                result_file.write('\n%s (OPT_PARAM)\n' % kernel.upper())
                clf = svm.SVC(kernel = kernel)
                
                for strat_kfold in STRAT_KFOLD_VALUES: 
                    if strat_kfold:
                        print ('%s with %s (OPT_PARAM) kernel and strat k-fold '
                               'CV on %s\n') % (embedding_name.upper(),
                                                kernel.upper(), dataset)
                    else:
                        print ('%s with %s (OPT_PARAM) kernel and k-fold CV on '
                               '%s\n') % (embedding_name.upper(), kernel.upper(),
                                          dataset)
                                          
                    
                    perform_eval.cross_val_opt_embedding_param(clf, graph_of_num,
                                                               embedding,
                                                               param_range,
                                                               strat_kfold,
                                                               NUM_IT, NUM_FOLDS,
                                                               result_file)                                           
        if not OPT and not COMPARE_PARAM:
            result_file.close()
            continue
        
        
        if OPT_PARAM:
            result_file.write('\n')
                
        for embedding_param in EMBEDDING_PARAMS.get(embedding_name, [None]):
            # --------------------------------------------------------------------
            # 2) extract features
            # --------------------------------------------------------------------
            data_matrix, class_lbls = extract_features(graph_of_num, embedding,
                                                       embedding_param, 
                                                       result_file)
            # !!
#            Z = data_matrix.todense()
    
    
            # --------------------------------------------------------------------
            # 3) evaluate performance
            # --------------------------------------------------------------------                                                                
            if COMPARE_PARAM:
                for kernel in KERNELS:
                    # initialize classifier
                    clf = svm.SVC(kernel = kernel)
                    result_file.write('\n%s\n' % kernel.upper())
                    
                    for strat_kfold in STRAT_KFOLD_VALUES:            
                        if strat_kfold:
                            print ('%s with %s kernel and strat k-fold CV on '
                                   '%s\n') % (embedding_name.upper(),
                                              kernel.upper(), dataset)
                        else:
                            print '%s with %s kernel and k-fold CV on %s\n' %\
                                  (embedding_name.upper(), kernel.upper(),
                                   dataset)
                
                        
                        perform_eval.cross_val(clf, data_matrix, class_lbls,
                                               NUM_IT, NUM_FOLDS, strat_kfold,
                                               result_file)
                                               
#            if OPT:
#                for strat_kfold in STRAT_KFOLD_VALUES:
#                    if strat_kfold:
#                        print ('%s with LINEAR/RBF kernel and strat k-fold CV '
#                               'on '%s\n') % (embedding_name.upper(), dataset)
#                    else:
#                        print ('%s with LINEAR/RBF kernel and k-fold CV on '
#                               '%s\n') % (embedding_name.upper(), dataset)
#                              
#                    perform_eval.cross_val_with_opt_clf(data_matrix, class_lbls,
#                                                        num_it = NUM_IT,
#                                                        ref_clf = None,
#                                                        strat_kfold =\
#                                                                     strat_kfold,
#                                                        verbose = False,
#                                                        result_file =\
#                                                                     result_file)
            
        result_file.close()

end_time = time.time()
total_time = end_time - start_time

print 'The evaluation of the emedding method(s) took %.1f seconds' % total_time




    
    

#        if opt:
#            # initialize classifier
#    #        ref_clf = svm.SVC(kernel = 'linear')
#    #        ref_clf = svm.SVC(kernel = 'poly')
#    #        ref_clf = svm.SVC(kernel = 'rbf') # default
#    #        ref_clf = svm.SVC(kernel = 'sigmoid')
#             # only 0.83 score on MUTAG, needs dense data_matrix and parameter
#             # with_mean set to True
#    #        ref_clf = make_pipeline(StandardScaler(with_mean = True), svm.SVC())
#
#    #        param_grid = [{'C': [1, 10, 100], 'kernel' : ['linear']}]
#    #        param_grid = [{'C': [1, 10, 100], 'kernel' : ['rbf']}]
#
#            lin_clf_scores, opt_clf_scores, cross_val_time =\
#            perform_eval.perform_cross_val_with_opt_clf(data_matrix,
#                                                        class_lbls,
#                                                        num_it = 1,
##                                                        param_grid = param_grid,
##                                                        ref_clf = ref_clf,
#                                                        ref_clf = None,
##                                                        strat_kfold = False,
#                                                        strat_kfold = True,
#                                                        verbose = False)
