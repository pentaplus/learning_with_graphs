"""
Evaluation of embedding methods.

This module provides functions for evaluating the performance of four
explecit two implicit graph embedding methods. The explicit ones are the
Weisfeiler-Lehman subtree kernel, the neighborhood hash kernel (in three
variants) and the !!. The implicit embeddings comprise the random walk
kernel and the !!. The classification accuracies and runtimes are
evaluated on the following 8 datasets: MUTAG, PTC(MR), ENZYMES, DD,
NCI1, NCI109, FLASH CFG, and ANDROID FCG.
"""

__author__ = "Benjamin Plock"
__date__ = "2016-02-28"


# planed procedure:
# at Benny-Notebook:
# 0. test WL on first 7 datasets (param range = {0,...,5})
# 1. test NH and CSNH all iter (10 iterations, libsvm-liblinear)
# 2. test GRAPHLET_KERNEL for param = 3 (10 iterations, libsvm-liblinear)
# 3. test GRAPHLET_KERNEL for param = 4 (10 iterations, libsvm-liblinear)
# 4. test WL on ANDROID FCG PARTIAL (10 iterations, LIBSVM)
# 
# at Ben-PC:
# 1. implement Eigen kernel
# 10. compress FCGs
# 
#
# 11. evaluate performance on ANDROID FCG PARTIAL
# 12. gradually increment the number of samples of ANDROID FCG PARTIAL
# 
# 
# 100. make feature vectors for NHGK unary 



import importlib
import inspect
import sys
import time

from os.path import abspath, dirname, join
from sklearn import svm
from sklearn.grid_search import GridSearchCV
#from sklearn.metrics.pairwise import pairwise_kernels
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler


# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))


from misc import dataset_loader, utils
from performance_evaluation import cross_validation


#=================================================================================
# constants
#=================================================================================

DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', 'datasets')

# embeddings
WEISFEILER_LEHMAN = 'weisfeiler_lehman'
NEIGHBORHOOD_HASH = 'neighborhood_hash'
COUNT_SENSITIVE_NEIGHBORHOOD_HASH = 'count_sensitive_neighborhood_hash'
COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER =\
                                      'count_sensitive_neighborhood_hash_all_iter'
GRAPHLET_KERNEL_3 = 'graphlet_kernel_3'
GRAPHLET_KERNEL_4 = 'graphlet_kernel_4'
LABEL_COUNTER = 'label_counter'
RANDOM_WALK_KERNEL = 'random_walk_kernel'

# datasets
MUTAG = 'MUTAG'
PTC_MR = 'PTC(MR)'
ENZYMES = 'ENZYMES'
DD = 'DD'
NCI1 = 'NCI1'
NCI109 = 'NCI109'
ANDROID_FCG_PARTIAL = 'ANDROID FCG PARTIAL'
FLASH_CFG = 'FLASH CFG'


#=================================================================================
# parameter definitions
#=================================================================================

#EMBEDDING_NAMES = [LABEL_COUNTER]
#EMBEDDING_NAMES = [WEISFEILER_LEHMAN, LABEL_COUNTER]
#EMBEDDING_NAMES = [WEISFEILER_LEHMAN]
#EMBEDDING_NAMES = [WEISFEILER_LEHMAN, COUNT_SENSITIVE_NEIGHBORHOOD_HASH]
#EMBEDDING_NAMES = [WEISFEILER_LEHMAN, COUNT_SENSITIVE_NEIGHBORHOOD_HASH,
#                   COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER]
#EMBEDDING_NAMES = [WEISFEILER_LEHMAN, NEIGHBORHOOD_HASH]
#EMBEDDING_NAMES = [COUNT_SENSITIVE_NEIGHBORHOOD_HASH]
#EMBEDDING_NAMES = [COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER]
#EMBEDDING_NAMES = [NEIGHBORHOOD_HASH, COUNT_SENSITIVE_NEIGHBORHOOD_HASH]
#EMBEDDING_NAMES = [GRAPHLET_KERNEL_3]
#EMBEDDING_NAMES = [GRAPHLET_KERNEL_4]
EMBEDDING_NAMES = [GRAPHLET_KERNEL_3, GRAPHLET_KERNEL_4]
#EMBEDDING_NAMES = [RANDOM_WALK_KERNEL]


# keys are indices of the list EMBEDDING_NAMES, values are the respective
# parameters
EMBEDDING_PARAM_RANGES = {
                          WEISFEILER_LEHMAN: range(6),
                          NEIGHBORHOOD_HASH: range(6),
                          COUNT_SENSITIVE_NEIGHBORHOOD_HASH: range(6),
                          COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER: range(6),
                          GRAPHLET_KERNEL_3: [None],
                          GRAPHLET_KERNEL_4: [None],
                          RANDOM_WALK_KERNEL: [None]
                         }

#DATASET = ANDROID_FCG_PARTIAL # !! increase number of samples

# sorted by number of graphs in ascending order
#DATASETS = [MUTAG, PTC_MR, ENZYMES, DD, NCI1, NCI109]
#DATASETS = [MUTAG, PTC_MR, ENZYMES, DD, NCI1, NCI109, FLASH_CFG]
DATASETS = [MUTAG, PTC_MR, ENZYMES]
#DATASETS = [DD, NCI1, NCI109]
#DATASETS = [MUTAG]
#DATASETS = [PTC_MR]
#DATASETS = [ENZYMES]
#DATASETS = [DD]
#DATASETS = [NCI1]
#DATASETS = [NCI109]
#DATASETS = [ANDROID_FCG_PARTIAL]
#DATASETS = [FLASH_CFG]

#OPT_PARAM = True
OPT_PARAM = False

#COMPARE_PARAMS = True
COMPARE_PARAMS = False

#SEARCH_OPT_SVM_PARAM_IN_PAR = True
SEARCH_OPT_SVM_PARAM_IN_PAR = False

NUM_ITER = 10
#NUM_ITER = 3
#NUM_ITER = 1

# maximum number of iterations for small datasets (i.e., less than 1000 samples)
MAX_ITER_SD = 1e7
#MAX_ITER_SD = -1

# maximum number of iterations for large datasets (i.e., more than 1000 samples)
MAX_ITER_LD = 1e3

# number of folds used in cross validation for performance evaluation
NUM_OUTER_FOLDS = 10

# number of folds used in cross validation on training data for small datasets
# (i.e., less than 1000 samples)
NUM_INNER_FOLDS_SD = 3

# number of folds used in cross validation on training data for large datasets
# (i.e., more than 1000 samples)
NUM_INNER_FOLDS_LD = 2



def extract_features(graph_meta_data_of_num, embedding, param_range, result_file):
    print '-------------------------------------------------------------\n'
    result_file.write('------------------------------------------\n\n')

    feat_extr_start_time = time.time()

    data_mat_of_param, extr_time_of_param =\
                   embedding.extract_features(graph_meta_data_of_num, param_range)

    feat_extr_end_time = time.time()
    feat_extr_time = feat_extr_end_time - feat_extr_start_time
    utils.write('Graph loading and feature exraction took %.1f seconds.\n' %\
                                                      feat_extr_time, result_file)
    print ''

    return data_mat_of_param, extr_time_of_param
    
    
def compute_kernel_matrix(graph_meta_data_of_num, embedding, param_range,
                          result_file):
    print '-------------------------------------------------------------\n'
    result_file.write('------------------------------------------\n\n')

    kernel_mat_comp_start_time = time.time()

    kernel_mat_of_param, kernel_mat_comp_time_of_param =\
                           embedding.compute_kernel_mat(graph_meta_data_of_num,
                                                        param_range)

    kernel_mat_comp_end_time = time.time()
    kernel_mat_comp_time = kernel_mat_comp_end_time - kernel_mat_comp_start_time
    utils.write('The computation of the kernel matrix took %.1f seconds.\n' %\
                                                kernel_mat_comp_time, result_file)
    print ''

    return kernel_mat_of_param, kernel_mat_comp_time_of_param
    
  
def init_clf(embedding_is_implicit, dataset_is_large, num_inner_folds):
    """
    Initialize classifier.
    
    For multiclass classification the One-Versus-Rest scheme is applied,
    i.e., in case of N different classes N classifiers are trained in
    total.
    """
    if dataset_is_large:
        max_iter = MAX_ITER_LD
        svm_param_grid = {'C': (0.01, 0.1, 1)}
        num_jobs = 3
    else:
        max_iter = MAX_ITER_SD
        svm_param_grid = {'kernel': ('linear', 'rbf'), 'C': (0.1, 10)}
        num_jobs = 4
    
    if embedding_is_implicit:
        # library LIBSVM is used
        clf = svm.SVC(kernel = 'precomputed', max_iter = max_iter,
                      decision_function_shape = 'ovr')
    elif dataset_is_large:
        # library LIBLINEAR is used
        clf = svm.LinearSVC(max_iter = max_iter)
    else:
        # library LIBSVM is used
        clf = svm.SVC(max_iter = max_iter, decision_function_shape = 'ovr')
    
    if SEARCH_OPT_SVM_PARAM_IN_PAR:
        grid_clf = GridSearchCV(clf, svm_param_grid, cv = num_inner_folds,
                                n_jobs = num_jobs, pre_dispatch = '2*n_jobs')
    else:
        grid_clf = GridSearchCV(clf, svm_param_grid, cv = num_inner_folds)
    
    return grid_clf
        

def set_params(dataset_is_large, embedding_is_implicit):
    num_inner_folds = NUM_INNER_FOLDS_LD if dataset_is_large\
                                                           else NUM_INNER_FOLDS_SD
    if embedding_is_implicit:
        # use library LIBSVM
        use_liblinear = False
        kernel = 'precomputed'
    else:
        # params for explicit embeddings
        if dataset_is_large:
            # use library LIBLINEAR
            use_liblinear = True
            kernel = 'linear'
        else:
            # use library LIBSVM
            use_liblinear = False
            kernel = 'linear/rbf'
        
    return use_liblinear, kernel, num_inner_folds
    

def is_embedding_implicit(embedding_name):
    implicit_embeddings = [RANDOM_WALK_KERNEL]
    return True if embedding_name in implicit_embeddings else False
    

def write_param_info(use_liblinear, embedding_is_implicit, num_inner_folds,
                     result_file):
    if use_liblinear:
        utils.write('LIBRARY: LIBLINEAR\n', result_file)
    else:
        utils.write('LIBRARY: LIBSVM\n', result_file)
    if embedding_is_implicit:
        utils.write('EMBEDDING TYPE: IMPLICIT\n', result_file)
    else:
        utils.write('EMBEDDING TYPE: EXPLICIT\n', result_file) 
    utils.write('SEARCH_OPT_SVM_PARAM_IN_PAR: %s\n' %\
                       SEARCH_OPT_SVM_PARAM_IN_PAR.__str__().upper(), result_file)
    utils.write('NUM_ITER: %d\n' % NUM_ITER, result_file)
    utils.write('NUM_OUTER_FOLDS: %d\n' % NUM_OUTER_FOLDS, result_file)
    if OPT_PARAM:
        utils.write('NUM_INNER_FOLDS: %d\n' % num_inner_folds, result_file)
    if not use_liblinear:
        if MAX_ITER_SD == -1:
            utils.write('MAX_ITER_SD: UNLIMITED\n', result_file)
        else:
            utils.write('MAX_ITER_SD: %.e\n' % MAX_ITER_SD, result_file)
    sys.stdout.write('\n')
    
    
def write_eval_info(dataset, embedding_name, kernel, mode = None):
    mode_str = ' (' + mode + ')' if mode else ''
    
    print ('%s with %s kernel%s on %s\n') %\
               (embedding_name.upper(), kernel.upper(), mode_str.upper(), dataset)
           

def write_extr_time_for_param(param, extr_time_of_param, result_file):
    print '-------------------------------------------------------------'
    result_file.write('------------------------------------------\n')
    utils.write('Parameter: %r\n\n' % param, result_file)
    utils.write('Feature extraction took %.1f seconds.\n' %\
                extr_time_of_param[param], result_file)
    sys.stdout.write('\n')
    
    
def write_kernel_comp_for_param(param, kernel_mat_comp_time_of_param,
                                result_file):
    print '-------------------------------------------------------------'
    result_file.write('------------------------------------------\n')
    utils.write('Parameter: %r\n\n' % param, result_file)
    utils.write('The computation of the kernel matrix took %.1f seconds.\n' %\
                kernel_mat_comp_time_of_param[param], result_file)
    sys.stdout.write('\n')
    
    

script_exec_start_time = time.time()

for dataset in DATASETS:
    #=============================================================================
    # 1) retrieve graph meta data and class lables
    #=============================================================================
    graph_meta_data_of_num, class_lbls =\
      dataset_loader.get_graph_meta_data_of_num_dict_and_class_lbls(dataset,
                                                                    DATASETS_PATH)
    
    num_samples = len(graph_meta_data_of_num)
    dataset_is_large = True if num_samples >= 1000 else False


    for embedding_name in EMBEDDING_NAMES:
        # set parameters depending on whether or not the number of samples within 
        # the dataset is larger than 1000 and depending on wether the embedding is
        # implict or explicit
        embedding = importlib.import_module('embeddings.' + embedding_name)
        embedding_is_implicit = is_embedding_implicit(embedding_name)
        
        use_liblinear, kernel, num_inner_folds =\
                               set_params(dataset_is_large, embedding_is_implicit)  
        
        param_range = EMBEDDING_PARAM_RANGES[embedding_name]
        
        result_path = join(SCRIPT_FOLDER_PATH, '..', 'results', embedding_name)
        utils.makedir(result_path)
        result_file = open(join(result_path, dataset + '.txt'), 'w')
        
        write_param_info(use_liblinear, embedding_is_implicit, num_inner_folds,
                         result_file)

        #=========================================================================
        # 2) extract features if embedding is an explicit embedding, else compute
        #    the kernel matrix
        #=========================================================================
        if not embedding_is_implicit:
            data_mat_of_param, extr_time_of_param =\
                  extract_features(graph_meta_data_of_num, embedding, param_range,
                                   result_file)
        else:
            kernel_mat_of_param, kernel_mat_comp_time_of_param =\
                          compute_kernel_matrix(graph_meta_data_of_num, embedding,
                                                param_range, result_file)
                                                
        # initialize SVM classifier
        grid_clf = init_clf(embedding_is_implicit, dataset_is_large,
                            num_inner_folds)
        
        if OPT_PARAM and len(param_range) > 1:
            #=====================================================================
            # 3) evaluate the embedding's performance with optimized embedding
            #    parameter (this is only done for explicit embeddings)
            #=====================================================================
            mode = 'opt_param'
            
            result_file.write('\n%s (%s)\n' % (kernel.upper(), mode.upper()))
            
            write_eval_info(dataset, embedding_name, kernel, mode)
            
            cross_validation.optimize_embedding_param(grid_clf, data_mat_of_param,
                                                      class_lbls, NUM_ITER,
                                                      NUM_OUTER_FOLDS,
                                                      num_inner_folds,
                                                      result_file)                                           
        if not COMPARE_PARAMS:
            result_file.close()
            continue
        
        
        if OPT_PARAM:
            result_file.write('\n')
                
                                                                
        if COMPARE_PARAMS:
            #=====================================================================
            # 4) evaluate the embedding's performance for each embedding
            #    parameter
            #=====================================================================
            for param in param_range:
                if not embedding_is_implicit:
                    write_extr_time_for_param(param, extr_time_of_param,
                                              result_file)
                else:
                    write_kernel_comp_for_param(param,
                                                kernel_mat_comp_time_of_param,
                                                result_file)
                
                result_file.write('\n%s\n' % kernel.upper())
                
                write_eval_info(dataset, embedding_name, kernel)
                
                if not embedding_is_implicit:
                    data_mat = data_mat_of_param[param]
                    cross_validation.cross_val(grid_clf, data_mat, class_lbls,
                                               NUM_ITER, NUM_OUTER_FOLDS,
                                               result_file)
                else:
#                    kernel_mat == data_mat.dot(data_mat.T) # !!
#                    kernel_mat = pairwise_kernels(data_mat)
                    kernel_mat = kernel_mat_of_param[param]
                    cross_validation.cross_val(grid_clf, kernel_mat, class_lbls,
                                               NUM_ITER, NUM_OUTER_FOLDS,
                                               result_file)

            
        result_file.close()

script_exec_end_time = time.time()
script_exec_time = script_exec_end_time - script_exec_start_time

print '\nThe evaluation of the emedding method(s) took %.1f seconds.' %\
                                                                  script_exec_time




#import numpy as np
#np.savetxt('P6.txt', data_matrices[5].todense())
#
#import numpy as np
#np.savetxt('N6.txt', data_mat.todense())
#def foo(x,y,z):
#    if x == 0:
#        print 'x'
#    elif y == 0:
#        print 'y'
#    else:
#        print 'z'