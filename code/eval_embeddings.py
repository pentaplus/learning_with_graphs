# planed procedure:
# at Benny-Notebook:
# 0. test WL on first 6 datasets (param range = {0,...5})
# 0. test WL on ANDROID FCG PARTIAL (10 iterations, LIBSVM)
# 1. test NH and CSNH all iter (10 iterations, libsvm-liblinear)
# 2. test GRAPHLET_KERNEL for param = 3 (10 iterations, libsvm-liblinear)
# 3. test GRAPHLET_KERNEL for param = 4 (10 iterations, libsvm-liblinear)
# 
# at Ben-PC:
# 1. optimize function optimize_embedding_param by a large factor
# 2. implement an implicit embedding method
# 4. compress FCGs
# 
#
# 4. evaluate performance on ANDROID FCG PARTIAL
# 5. gradually increment the number of samples of ANDROID FCG PARTIAL
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
CFG = 'CFG'

#EMBEDDING_NAMES = [LABEL_COUNTER]
#EMBEDDING_NAMES = [WEISFEILER_LEHMAN, LABEL_COUNTER]
#EMBEDDING_NAMES = [WEISFEILER_LEHMAN]
EMBEDDING_NAMES = [WEISFEILER_LEHMAN, COUNT_SENSITIVE_NEIGHBORHOOD_HASH]
#EMBEDDING_NAMES = [WEISFEILER_LEHMAN, NEIGHBORHOOD_HASH]
#EMBEDDING_NAMES = [COUNT_SENSITIVE_NEIGHBORHOOD_HASH]
#EMBEDDING_NAMES = [COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER]
#EMBEDDING_NAMES = [NEIGHBORHOOD_HASH, COUNT_SENSITIVE_NEIGHBORHOOD_HASH]
#EMBEDDING_NAMES = [GRAPHLET_KERNEL_3]
#EMBEDDING_NAMES = [GRAPHLET_KERNEL_4]


# keys are indices of the list EMBEDDING_NAMES, values are the respective
# parameters
EMBEDDING_PARAM_RANGES = {
#                         WEISFEILER_LEHMAN : [10],
                          WEISFEILER_LEHMAN : [0, 1, 2, 3, 4, 5],
                          NEIGHBORHOOD_HASH : [0, 1, 2, 3, 4, 5],
                          COUNT_SENSITIVE_NEIGHBORHOOD_HASH : [0, 1, 2, 3, 4, 5],
                          COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER :\
                                                               [0, 1, 2, 3, 4, 5],
                          GRAPHLET_KERNEL_3 : [3],
                          GRAPHLET_KERNEL_4 : [4],
                          RANDOM_WALK_KERNEL: [None]
                         }


#DATASET = 'ANDROID FCG' # !! change file names from hashes to numbers
#DATASET = 'CFG' # !! change file names from hashes to numbers

# sorted by number of graphs in ascending order
DATASETS = [MUTAG, PTC_MR, ENZYMES, DD, NCI1, NCI109]
#DATASETS = [MUTAG, PTC_MR, ENZYMES]
#DATASETS = [DD, NCI1, NCI109]
#DATASETS = [MUTAG]
#DATASETS = [PTC_MR]
#DATASETS = [ENZYMES]
#DATASETS = [DD]
#DATASETS = [NCI1]
#DATASETS = [NCI109]
#DATASETS = [ANDROID_FCG_PARTIAL]
#DATASETS = [CFG]

OPT_PARAM = True
#OPT_PARAM = False

COMPARE_PARAMS = True
#COMPARE_PARAMS = False

#OPT = True
OPT = False

# kernels for LIBSVM classifier
#LIBSVM_KERNELS = ['linear', 'rbf', 'poly', 'sigmoid']
#LIBSVM_KERNELS = ['linear', 'rbf', 'sigmoid']
LIBSVM_KERNELS = ['linear', 'rbf']
#LIBSVM_KERNELS = ['linear']
#LIBSVM_KERNELS = ['rbf']
#LIBSVM_KERNELS = ['sigmoid']
#LIBSVM_KERNELS = ['poly']

#STRAT_KFOLD_VALUES = [False, True]
STRAT_KFOLD_VALUES = [False]
#STRAT_KFOLD_VALUES = [True]

#NUM_ITER = 1
NUM_ITER = 10

NUM_FOLDS = 10

NUM_INNER_FOLDS_SD = 10
#NUM_INNER_FOLDS_SD = 4 # !!
NUM_INNER_FOLDS_LD = 4

LIMIT_CLF_MAX_ITER_SD = False
#LIMIT_CLF_MAX_ITER_SD = True
LIMIT_CLF_MAX_ITER_LD = False
#LIMIT_CLF_MAX_ITER_LD = True



def load_dataset(dataset, datasets_path):
    dataset_loading_start_time = time.time()

    graph_of_num, class_lbls = dataset_loader.load_dataset(datasets_path, dataset)

    dataset_loading_end_time = time.time()
    dataset_loading_time = dataset_loading_end_time - dataset_loading_start_time
    print 'Loading the dataset %s took %.1f seconds.\n' % (dataset,
                                                           dataset_loading_time)
    return graph_of_num, class_lbls
    

def extract_features(graph_of_num, embedding, param_range, result_file):
    sys.stdout.write(('----------------------------------------------------------'
                      '---\n\n'))
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
    
    
def init_clf(liblinear, max_iter, kernel = None):
    # !!
#    return svm.SVC(kernel = kernel, decision_function_shape = 'ovr',
#                   max_iter = max_iter)     
#    return svm.LinearSVC(max_iter = max_iter) # !!
    
    if LIBLINEAR:
        # library LIBLINEAR is used
        # for multiclass classification the One-Versus-Rest scheme is applied,
        # i.e., in case of N different classes N classifiers are trained in total
        return svm.LinearSVC(max_iter = max_iter)
    else:
        # library LIBSVM is used
        # for multiclass classification also the One-Versus-Rest scheme is applied
        return svm.SVC(kernel = kernel, decision_function_shape = 'ovr',
                       max_iter = max_iter)
        

def set_params(num_samples, dataset, limit_clf_max_iter_sd,
               limit_clf_max_iter_ld):
    if num_samples > 1000:
        # use library LIBLINEAR
        LIBLINEAR = True
        KERNELS = ['linear']
        NUM_INNER_FOLDS = NUM_INNER_FOLDS_LD
        CLF_MAX_ITER = 100 if LIMIT_CLF_MAX_ITER_LD else 1000
    else:
        # use library LIBSVM
        LIBLINEAR = False
        KERNELS = LIBSVM_KERNELS
        NUM_INNER_FOLDS = NUM_INNER_FOLDS_SD
        CLF_MAX_ITER = 100 if LIMIT_CLF_MAX_ITER_SD else -1
        
    return LIBLINEAR, KERNELS, NUM_INNER_FOLDS, CLF_MAX_ITER
    

def write_param_info(liblinear, num_iter, opt_param, num_inner_folds,
                     clf_max_iter, result_file):
    if liblinear:
        utils.write('LIBRARY: LIBLINEAR\n', result_file)
    else:
        utils.write('LIBRARY: LIBSVM\n', result_file)
    utils.write('NUM_ITER: %d\n' % num_iter, result_file)
    if opt_param:
        utils.write('NUM_INNER_FOLDS: %d\n' % num_inner_folds, result_file)
        if clf_max_iter == -1:
            utils.write('CLF_MAX_ITER: -1 (unlimited)\n', result_file)
        else:
            utils.write('CLF_MAX_ITER: %d\n' % clf_max_iter, result_file)
    print ''
    
    
def write_eval_info(dataset, embedding_name, kernel, strat_kfold, mode = None):
    mode_str = ' (' + mode + ')' if mode else ''
    kfold_str = 'strat k-fold' if strat_kfold else 'k-fold'
    
    print ('%s with %s kernel%s and %s CV on %s\n') %\
          (embedding_name.upper(), kernel.upper(), mode_str.upper(), kfold_str,
           dataset)
           

def write_extr_time_for_param(param, extr_time_of_param, result_file):
    sys.stdout.write(('----------------------------------------------------------'
                      '---\n'))
    result_file.write('------------------------------------------\n')
    utils.write('Parameter: %d\n\n' % param, result_file)
    utils.write('Feature extraction took %.1f seconds.\n' %\
                extr_time_of_param[param], result_file)
    sys.stdout.write('\n')
    

script_exec_start_time = time.time()

for dataset in DATASETS:
    # ----------------------------------------------------------------------------
    # 1) load dataset
    # ----------------------------------------------------------------------------
    graph_of_num, class_lbls = load_dataset(dataset, DATASETS_PATH)
    
    num_samples = len(graph_of_num)
    
    # set parameters depending on whether or not the number of samples within the
    # dataset is larger than 1000
    LIBLINEAR, KERNELS, NUM_INNER_FOLDS, CLF_MAX_ITER =\
    set_params(num_samples, dataset, LIMIT_CLF_MAX_ITER_SD, LIMIT_CLF_MAX_ITER_LD)
        
    
    for embedding_name in EMBEDDING_NAMES:
        result_path = join(SCRIPT_FOLDER_PATH, '..', 'results', embedding_name)
        utils.makedir(result_path)
        result_file = open(join(result_path, dataset + '.txt'), 'w')
        
        write_param_info(LIBLINEAR, NUM_ITER, OPT_PARAM, NUM_INNER_FOLDS,
                         CLF_MAX_ITER, result_file)
        
        embedding = importlib.import_module('embeddings.' + embedding_name)
        
        param_range = EMBEDDING_PARAM_RANGES[embedding_name]
        
        #-------------------------------------------------------------------------
        # 2) extract features
        # ------------------------------------------------------------------------
        data_mat_of_param, extr_time_of_param =\
               extract_features(graph_of_num, embedding, param_range, result_file)
        
        if OPT_PARAM:
            #---------------------------------------------------------------------
            # 3) evaluate the embedding's performance with optimized embedding
            #    parameter
            # --------------------------------------------------------------------
            if len(param_range) == 1:
                print ('The embedding ' + embedding_name + ' has only one '
                       'parameter. Therefore, its evaluation with optimized '
                       'embedding parameter is skipped.')
                continue
        
            mode = 'opt_param'
            
            for kernel in KERNELS:
                result_file.write('\n%s (%s)\n' % (kernel.upper(), mode.upper()))
                
                # initialize SVM classifier
                clf = init_clf(LIBLINEAR, CLF_MAX_ITER, kernel)
                
                for strat_kfold in STRAT_KFOLD_VALUES:
                    write_eval_info(dataset, embedding_name, kernel, strat_kfold,
                                    mode)
                    
                    cross_validation.optimize_embedding_param(clf,
                                                              data_mat_of_param,
                                                              class_lbls,
                                                              strat_kfold,
                                                              NUM_ITER,
                                                              NUM_FOLDS,
                                                              NUM_INNER_FOLDS,
                                                              result_file)                                           
        if not OPT and not COMPARE_PARAMS:
            result_file.close()
            continue
        
        
        if OPT_PARAM:
            result_file.write('\n')
                
                                                                
        if COMPARE_PARAMS:
            # --------------------------------------------------------------------
            # 4) evaluate the embedding's performance for each embedding
            #    parameter
            # --------------------------------------------------------------------
            for param, data_mat in data_mat_of_param.iteritems():
                write_extr_time_for_param(param, extr_time_of_param, result_file)                
                
                for kernel in KERNELS:
                    # initialize SVM classifier
                    clf = init_clf(LIBLINEAR, CLF_MAX_ITER, kernel)
                    
                    result_file.write('\n%s\n' % kernel.upper())
                    
                    for strat_kfold in STRAT_KFOLD_VALUES:
                        
                
                        write_eval_info(dataset, embedding_name, kernel,
                                        strat_kfold)
                        
                        cross_validation.cross_val(clf, data_mat, class_lbls,
                                                   NUM_ITER, NUM_FOLDS,
                                                   strat_kfold, result_file)
# !!                                               
#            if OPT:
#                for strat_kfold in STRAT_KFOLD_VALUES:
#                    if strat_kfold:
#                        print ('%s with LINEAR/RBF kernel and strat k-fold CV '
#                               'on '%s\n') % (embedding_name.upper(), dataset)
#                    else:
#                        print ('%s with LINEAR/RBF kernel and k-fold CV on '
#                               '%s\n') % (embedding_name.upper(), dataset)
#                              
#                    cross_validation.optimize_gen_params(data_mat, class_lbls,
#                                                         num_iter = NUM_ITER,
#                                                         ref_clf = None,
#                                                         strat_kfold =\
#                                                                     strat_kfold,
#                                                         verbose = False,
#                                                         result_file =\
#                                                                     result_file)
            
        result_file.close()

script_exec_end_time = time.time()
script_exec_time = script_exec_end_time - script_exec_start_time

print '\nThe evaluation of the emedding method(s) took %.1f seconds.' %\
                                                                  script_exec_time


#result_file = open('test', 'w')
#
#for embedding_param in [10]:
#    # --------------------------------------------------------------------
#    # 2) extract features
#    # --------------------------------------------------------------------
#    data_mat, class_lbls = extract_features(graph_of_num, embedding,
#                                               embedding_param, 
#                                               result_file)

    
    
# !!
#        if opt:
#    #        ref_clf = svm.SVC(kernel = 'linear')
#    #        ref_clf = svm.SVC(kernel = 'poly')
#    #        ref_clf = svm.SVC(kernel = 'rbf') # default
#    #        ref_clf = svm.SVC(kernel = 'sigmoid')
#             # only 0.83 score on MUTAG, needs dense data_mat and parameter
#             # with_mean set to True
#    #        ref_clf = make_pipeline(StandardScaler(with_mean = True), svm.SVC())
#
#    #        param_grid = [{'C': [1, 10, 100], 'kernel' : ['linear']}]
#    #        param_grid = [{'C': [1, 10, 100], 'kernel' : ['rbf']}]
#
#            lin_clf_scores, opt_clf_scores, cross_val_time =\
#            cross_validation.optimize_gen_params(data_mat,
#                                                 class_lbls,
#                                                 num_iter = 1,
##                                                 param_grid = param_grid,
##                                                 ref_clf = ref_clf,
#                                                 ref_clf = None,
##                                                 strat_kfold = False,
#                                                 strat_kfold = True,
#                                                 verbose = False)


#import numpy as np
#np.savetxt('P6.txt', data_matrices[5].todense())
#
#import numpy as np
#np.savetxt('N6.txt', data_mat.todense())