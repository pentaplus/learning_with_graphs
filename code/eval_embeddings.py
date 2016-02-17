# planed procedure:
# at Benny-Notebook:
# 0. test WL on first 6 datasets (param range = {0,...,10})
# 1. test NH and CSNH all iter (10 iterations, libsvm-liblinear)
# 2. test GRAPHLET_KERNEL for param = 3 (10 iterations, libsvm-liblinear)
# 3. test GRAPHLET_KERNEL for param = 4 (10 iterations, libsvm-liblinear)
# 4. test WL on ANDROID FCG PARTIAL (10 iterations, LIBSVM)
# 
# at Ben-PC:
# 1. implement an implicit embedding method
# 10. compress FCGs
# 
#
# 11. evaluate performance on ANDROID FCG PARTIAL
# 12. gradually increment the number of samples of ANDROID FCG PARTIAL
# 
# 
# 100. make feature vectors for NHGK unary 

# CAREFUL: PERFECTIONISM!
# 1000. make a grid search for kernels = ['linear', 'rbf'] in each optimization
# step (speedup by parallelization?)


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

# --------------------------------------------------------------------------------
# parameter definitions
# --------------------------------------------------------------------------------
DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', 'datasets')

# embeddings
WEISFEILER_LEHMAN = 'weisfeiler_lehman'
NEIGHBORHOOD_HASH = 'neighborhood_hash'
COUNT_SENSITIVE_NEIGHBORHOOD_HASH = 'count_sensitive_neighborhood_hash'
COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER =\
                                      'count_sensitive_neighborhood_hash_all_iter'
GRAPHLET_KERNEL = 'graphlet_kernel'
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
CFG = 'CFG'

#EMBEDDING_NAMES = [LABEL_COUNTER]
#EMBEDDING_NAMES = [WEISFEILER_LEHMAN, LABEL_COUNTER]
EMBEDDING_NAMES = [WEISFEILER_LEHMAN]
#EMBEDDING_NAMES = [WEISFEILER_LEHMAN, COUNT_SENSITIVE_NEIGHBORHOOD_HASH]
#EMBEDDING_NAMES = [WEISFEILER_LEHMAN, NEIGHBORHOOD_HASH]
#EMBEDDING_NAMES = [COUNT_SENSITIVE_NEIGHBORHOOD_HASH]
#EMBEDDING_NAMES = [COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER]
#EMBEDDING_NAMES = [NEIGHBORHOOD_HASH, COUNT_SENSITIVE_NEIGHBORHOOD_HASH]
#EMBEDDING_NAMES = [GRAPHLET_KERNEL]


# keys are indices of the list EMBEDDING_NAMES, values are the respective
# parameters
EMBEDDING_PARAM_RANGES = {
                          WEISFEILER_LEHMAN : range(6),
                          NEIGHBORHOOD_HASH : range(6),
                          COUNT_SENSITIVE_NEIGHBORHOOD_HASH : range(6),
                          COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER : range(6),                                   
                          GRAPHLET_KERNEL : [3],
#                          GRAPHLET_KERNEL : [4],
                          RANDOM_WALK_KERNEL: [None]
                         }

#DATASET = 'ANDROID FCG' # !! change file names from hashes to numbers
#DATASET = 'CFG' # !! change file names from hashes to numbers

# sorted by number of graphs in ascending order
#DATASETS = [MUTAG, PTC_MR, ENZYMES, DD, NCI1, NCI109]
#DATASETS = [MUTAG, PTC_MR, ENZYMES]
#DATASETS = [DD, NCI1, NCI109]
#DATASETS = [MUTAG]
#DATASETS = [PTC_MR]
#DATASETS = [ENZYMES]
DATASETS = [DD]
#DATASETS = [NCI1]
#DATASETS = [NCI109]
#DATASETS = [ANDROID_FCG_PARTIAL]
#DATASETS = [CFG]

OPT_PARAM = True
#OPT_PARAM = False

COMPARE_PARAMS = True
#COMPARE_PARAMS = False

#NUM_ITER = 10
NUM_ITER = 1

NUM_FOLDS = 10

NUM_INNER_FOLDS_SD = 10

NUM_INNER_FOLDS_LD = 4




def load_dataset(dataset, datasets_path):
    dataset_loading_start_time = time.time()

    graph_of_num, class_lbls = dataset_loader.load_dataset(datasets_path, dataset)

    dataset_loading_end_time = time.time()
    dataset_loading_time = dataset_loading_end_time - dataset_loading_start_time
    print 'Loading the dataset %s took %.1f seconds.\n' % (dataset,
                                                           dataset_loading_time)
    return graph_of_num, class_lbls
    

def extract_features(graph_of_num, embedding, param_range, result_file):
    print '-------------------------------------------------------------\n'
    result_file.write('------------------------------------------\n\n')

    feat_extr_start_time = time.time()

    data_mat_of_param, extr_time_of_param =\
                             embedding.extract_features(graph_of_num, param_range)

    feat_extr_end_time = time.time()
    feat_extr_time = feat_extr_end_time - feat_extr_start_time
    utils.write('Total feature extraction took %.1f seconds.\n' % feat_extr_time,
                result_file)
    print ''

    return data_mat_of_param, extr_time_of_param
    
    
def compute_kernel_matrix(graph_of_num, embedding, param_range, result_file):
    print '-------------------------------------------------------------\n'
    result_file.write('------------------------------------------\n\n')

    kernel_mat_comp_start_time = time.time()

    kernel_mat_of_param, kernel_mat_comp_time_of_param =\
                           embedding.compute_kernel_mat(graph_of_num, param_range)

    kernel_mat_comp_end_time = time.time()
    kernel_mat_comp_time = kernel_mat_comp_end_time - kernel_mat_comp_start_time
    utils.write('The computation of the kernel matrix took %.1f seconds.\n' %\
                                                kernel_mat_comp_time, result_file)
    print ''

    return kernel_mat_of_param, kernel_mat_comp_time_of_param
    
  
def init_clf(liblinear, embedding_is_implicit = False):
    # !!
#    return svm.SVC(kernel = kernel, decision_function_shape = 'ovr')     
#    return svm.LinearSVC() # !!
    if embedding_is_implicit:
        return svm.SVC(kernel = 'precomputed', decision_function_shape = 'ovr')
    
    if liblinear:
        # library LIBLINEAR is used
        # for multiclass classification the One-Versus-Rest scheme is applied,
        # i.e., in case of N different classes N classifiers are trained in total

#        svm_param_grid = {'kernel' : ('linear', 'rbf'), 'C' : [1, 10]}
#        grid_clf = GridSearchCV(svm.SVC(decision_function_shape = 'ovr'),
#                                svm_param_grid, cv = 3)
#        return grid_clf
    
        svm_param_grid = {'C' : [1, 10]}
        grid_clf = GridSearchCV(svm.LinearSVC(), svm_param_grid, cv = 3)
        return grid_clf
        
        
#        return svm.LinearSVC()
    else:
        # library LIBSVM is used
        # for multiclass classification also the One-Versus-Rest scheme is applied
        
#        svm_param_grid = {'kernel' : ['linear', 'rbf']}
        svm_param_grid = {'kernel' : ('linear', 'rbf'), 'C' : [1, 10]}
        grid_clf = GridSearchCV(svm.SVC(decision_function_shape = 'ovr'),
                                svm_param_grid, cv = 3)
                                
#        grid_clf = GridSearchCV(svm.SVC(decision_function_shape = 'ovr'),
#                                svm_param_grid, cv = 3, n_jobs = 4,
#                                pre_dispatch = '2*n_jobs')
                                
        return grid_clf
        

def set_params(num_samples, embedding_is_implicit):
    if num_samples > 1000:
        num_inner_folds = NUM_INNER_FOLDS_LD
    else:
        num_inner_folds = NUM_INNER_FOLDS_SD
        
    if embedding_is_implicit:
        # use library LIBSVM
        liblinear = False
        kernel = 'precomputed'
    else:
        # params for explicit embeddings
        if num_samples > 1000:
            # use library LIBLINEAR
            liblinear = True
            kernel = 'linear'
        else:
            # use library LIBSVM
            liblinear = False
            kernel = 'linear/rbf'
        
    return liblinear, kernel, num_inner_folds
    

def is_embedding_implicit(embedding_name):
    if embedding_name in [RANDOM_WALK_KERNEL]:
        return True
    else:
        return False
    

def write_param_info(liblinear, num_iter, opt_param, num_inner_folds,
                     result_file):
    if liblinear:
        utils.write('LIBRARY: LIBLINEAR\n', result_file)
    else:
        utils.write('LIBRARY: LIBSVM\n', result_file)
    utils.write('NUM_ITER: %d\n' % num_iter, result_file)
    if opt_param:
        utils.write('NUM_INNER_FOLDS: %d\n' % num_inner_folds, result_file)
    sys.stdout.write('\n')
    
    
def write_eval_info(dataset, embedding_name, kernel, mode = None):
    mode_str = ' (' + mode + ')' if mode else ''
    
    print ('%s with %s kernel%s on %s\n') %\
               (embedding_name.upper(), kernel.upper(), mode_str.upper(), dataset)
           

def write_extr_time_for_param(param, extr_time_of_param, result_file):
    print '-------------------------------------------------------------'
    result_file.write('------------------------------------------\n')
    utils.write('Parameter: %d\n\n' % param, result_file)
    utils.write('Feature extraction took %.1f seconds.\n' %\
                extr_time_of_param[param], result_file)
    sys.stdout.write('\n')
    
    
def write_kernel_comp_for_param(param, kernel_mat_comp_time_of_param,
                                result_file):
    print '-------------------------------------------------------------'
    result_file.write('------------------------------------------\n')
    utils.write('Parameter: %d\n\n' % param, result_file)
    utils.write('The computation of the kernel matrix took %.1f seconds.\n' %\
                kernel_mat_comp_time_of_param[param], result_file)
    sys.stdout.write('\n')
    
    

script_exec_start_time = time.time()

for dataset in DATASETS:
    # ----------------------------------------------------------------------------
    # 1) load dataset
    # ----------------------------------------------------------------------------
    graph_of_num, class_lbls = load_dataset(dataset, DATASETS_PATH)
    
    num_samples = len(graph_of_num)
    

    for embedding_name in EMBEDDING_NAMES:
        # set parameters depending on whether or not the number of samples within the
        # dataset is larger than 1000 and depending on wether the embedding is
        # implict or explicit
        embedding = importlib.import_module('embeddings.' + embedding_name)
        embedding_is_implicit = is_embedding_implicit(embedding_name)
        
        liblinear, kernel, num_inner_folds = set_params(num_samples,
                                                        embedding_is_implicit)  
        
        param_range = EMBEDDING_PARAM_RANGES[embedding_name]
        
        result_path = join(SCRIPT_FOLDER_PATH, '..', 'results', embedding_name)
        utils.makedir(result_path)
        result_file = open(join(result_path, dataset + '.txt'), 'w')
        
        write_param_info(liblinear, NUM_ITER, OPT_PARAM, num_inner_folds,
                         result_file)

        #-------------------------------------------------------------------------
        # 2) extract features if embedding is an explicit embedding, else compute
        #    the kernel matrix
        # ------------------------------------------------------------------------
        if not embedding_is_implicit:
            data_mat_of_param, extr_time_of_param =\
               extract_features(graph_of_num, embedding, param_range, result_file)
        else:
            kernel_mat_of_param, kernel_mat_comp_time_of_param =\
                       compute_kernel_matrix(graph_of_num, embedding, param_range,
                                             result_file)
        
        if OPT_PARAM and len(param_range) > 1:
            #---------------------------------------------------------------------
            # 3) evaluate the embedding's performance with optimized embedding
            #    parameter (this is only done for explicit embeddings)
            # --------------------------------------------------------------------
            mode = 'opt_param'
            
            
            result_file.write('\n%s (%s)\n' % (kernel.upper(), mode.upper()))
            
            # initialize SVM classifier
            clf = init_clf(liblinear)
            
            write_eval_info(dataset, embedding_name, kernel, mode)
            
            cross_validation.optimize_embedding_param(clf, data_mat_of_param,
                                                      class_lbls, NUM_ITER,
                                                      NUM_FOLDS, num_inner_folds,
                                                      result_file)                                           
        if not COMPARE_PARAMS:
            result_file.close()
            continue
        
        
        if OPT_PARAM:
            result_file.write('\n')
                
                                                                
        if COMPARE_PARAMS:
            # --------------------------------------------------------------------
            # 4) evaluate the embedding's performance for each embedding
            #    parameter
            # --------------------------------------------------------------------
        
#            for param, data_mat in data_mat_of_param.iteritems():
            for param in param_range:
                if not embedding_is_implicit:
                    write_extr_time_for_param(param, extr_time_of_param,
                                              result_file)
                else:
                    write_kernel_comp_for_param(param,
                                                kernel_mat_comp_time_of_param,
                                                result_file)
                
                
                # initialize SVM classifier
                clf = init_clf(liblinear, embedding_is_implicit)
                
                result_file.write('\n%s\n' % kernel.upper())
                
                write_eval_info(dataset, embedding_name, kernel)
                
                if not embedding_is_implicit:
                    data_mat = data_mat_of_param[param]
                    cross_validation.cross_val(clf, data_mat, class_lbls,
                                               NUM_ITER, NUM_FOLDS, result_file)
                else:
#                    kernel_mat == data_mat.dot(data_mat.T) # !!
#                    kernel_mat = pairwise_kernels(data_mat)
                    kernel_mat = kernel_mat_of_param[param]
                    cross_validation.cross_val(clf, kernel_mat, class_lbls,
                                               NUM_ITER, NUM_FOLDS, result_file)

            
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
