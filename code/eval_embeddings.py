# planed procedure:
# at night: run on all datasets with NUM_ITER = 10
# 
# 1. test on PTC(MR): 10 iter of LIBSVM, 10 iter of LIBLINEAR
#
# 1. implement neighborhood hash kernel
# 2. test h = 1 in WEISFEILER_LEHMAN


import importlib
import inspect
import time

from os.path import abspath, dirname, join
from sklearn import svm
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import StandardScaler


# determine script path
FILE_NAME = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_PATH = dirname(abspath(FILE_NAME))

# !!
del FILE_NAME

from misc import datasetloader, utils
from performance_evaluation import cross_validation

# --------------------------------------------------------------------------------
# parameter definitions
# --------------------------------------------------------------------------------
DATASETS_PATH = join(SCRIPT_PATH, '..', 'datasets')

WEISFEILER_LEHMAN = 'weisfeiler_lehman'
LABEL_COUNTER = 'label_counter'

#EMBEDDING_NAMES = [LABEL_COUNTER]
#EMBEDDING_NAMES = [WEISFEILER_LEHMAN, LABEL_COUNTER]
EMBEDDING_NAMES = [WEISFEILER_LEHMAN]

# keys are indices of the list EMBEDDING_NAMES, values are the respective
# parameters
EMBEDDING_PARAMS = {WEISFEILER_LEHMAN : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
#EMBEDDING_PARAMS = {WEISFEILER_LEHMAN : [0, 1, 2]}

#DATASET = 'ANDROID FCG' # !! change file names from hashes to numbers
#DATASET = 'CFG' # !! change file names from hashes to numbers

# sorted by number of graphs times in ascending order
DATASETS = ['MUTAG', 'PTC(MR)', 'ENZYMES', 'DD', 'NCI1', 'NCI109']
#DATASETS = ['ENZYMES', 'NCI109', 'NCI1', 'DD']
#DATASETS = ['MUTAG', 'PTC(MR)']
#DATASETS = ['NCI109']
#DATASETS = ['ENZYMES']
#DATASETS = ['MUTAG']
#DATASETS = ['DD']
#DATASETS = ['PTC(MR)']

OPT_PARAM = True
#OPT_PARAM = False

COMPARE_PARAM = True
#COMPARE_PARAM = False

#OPT = True
OPT = False

#KERNELS = ['linear', 'rbf', 'poly', 'sigmoid']
#KERNELS = ['linear', 'rbf', 'sigmoid']
#KERNELS = ['linear', 'rbf']
KERNELS = ['linear']

#STRAT_KFOLD_VALUES = [False, True]
STRAT_KFOLD_VALUES = [False]
#STRAT_KFOLD_VALUES = [True]

NUM_ITER = 1

NUM_FOLDS = 10

NUM_INNER_FOLDS_SD = 10
NUM_INNER_FOLDS_LD = 4 # even on DD 0.79 were reached!!! :-)

LIMIT_CLF_MAX_ITER_SD = False
#LIMIT_CLF_MAX_ITER_SD = True
LIMIT_CLF_MAX_ITER_LD = False
#LIMIT_CLF_MAX_ITER_LD = True



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

for dataset in DATASETS:
    # ----------------------------------------------------------------------------
    # 1) load dataset
    # ----------------------------------------------------------------------------
    graph_of_num = load_dataset(dataset, DATASETS_PATH)
    
    num_samples = len(graph_of_num)
    if num_samples > 1000:
        LIBLINEAR = True
        NUM_INNER_FOLDS = NUM_INNER_FOLDS_LD
        CLF_MAX_ITER = 100 if LIMIT_CLF_MAX_ITER_LD else 1000
    else:
        # use library LIBSVM
        LIBLINEAR = False
        NUM_INNER_FOLDS = NUM_INNER_FOLDS_SD
        CLF_MAX_ITER = 100 if LIMIT_CLF_MAX_ITER_SD else -1  
    
    
    for embedding_name in EMBEDDING_NAMES:
        result_path = join(SCRIPT_PATH, '..', 'results', embedding_name)
        utils.makedir(result_path)
        result_file = open(join(result_path, dataset + '.txt'), 'w')
        
        if LIBLINEAR:
            utils.write('LIBRARY: LIBLINEAR\n', result_file)
        else:
            utils.write('LIBRARY: LIBSVM\n', result_file)
        utils.write('NUM_ITER: %d\n' % NUM_ITER, result_file)
        if OPT_PARAM:
            utils.write('NUM_INNER_FOLDS: %d\n' % NUM_INNER_FOLDS, result_file)
            if CLF_MAX_ITER == -1:
                utils.write('CLF_MAX_ITER: -1 (unlimited)\n', result_file)
            else:
                utils.write('CLF_MAX_ITER: %d\n' % CLF_MAX_ITER, result_file)
        print ''
        
        embedding = importlib.import_module('embeddings.' + embedding_name)
        
        if OPT_PARAM:
            param_range = EMBEDDING_PARAMS[embedding_name]
            
            for kernel in KERNELS:
                result_file.write('\n%s (OPT_PARAM)\n' % kernel.upper())
                
                # initialize classifier
                if LIBLINEAR:
                    # library LIBLINEAR is used
                    clf = svm.LinearSVC(max_iter = CLF_MAX_ITER)
                else:
                    # library LIBSVM is used
                    clf = svm.SVC(kernel = kernel, max_iter = CLF_MAX_ITER)
                
                for strat_kfold in STRAT_KFOLD_VALUES: 
                    if strat_kfold:
                        print ('%s with %s (OPT_PARAM) kernel and strat k-fold '
                               'CV on %s\n') % (embedding_name.upper(),
                                                kernel.upper(), dataset)
                    else:
                        print ('%s with %s (OPT_PARAM) kernel and k-fold CV on '
                               '%s\n') % (embedding_name.upper(), kernel.upper(),
                                          dataset)
                    
                    cross_validation.optimize_embedding_param(clf, graph_of_num,
#                    cross_validation.optimize_embedding_and_kernel_param(
#                                                              graph_of_num,
                                                              embedding,
                                                              param_range,
                                                              strat_kfold,
                                                              NUM_ITER,
                                                              NUM_FOLDS,
                                                              NUM_INNER_FOLDS,
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
                
                        
                        cross_validation.cross_val(clf, data_matrix, class_lbls,
                                                   NUM_ITER, NUM_FOLDS,
                                                   strat_kfold, result_file)
                                               
#            if OPT:
#                for strat_kfold in STRAT_KFOLD_VALUES:
#                    if strat_kfold:
#                        print ('%s with LINEAR/RBF kernel and strat k-fold CV '
#                               'on '%s\n') % (embedding_name.upper(), dataset)
#                    else:
#                        print ('%s with LINEAR/RBF kernel and k-fold CV on '
#                               '%s\n') % (embedding_name.upper(), dataset)
#                              
#                    cross_validation.optimize_gen_params(data_matrix, class_lbls,
#                                                         num_iter = NUM_ITER,
#                                                         ref_clf = None,
#                                                         strat_kfold =\
#                                                                     strat_kfold,
#                                                         verbose = False,
#                                                         result_file =\
#                                                                     result_file)
            
        result_file.close()

end_time = time.time()
total_time = end_time - start_time

print 'The evaluation of the emedding method(s) took %.1f seconds' % total_time




    
    

#        if opt:
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
#            cross_validation.optimize_gen_params(data_matrix,
#                                                 class_lbls,
#                                                 num_iter = 1,
##                                                 param_grid = param_grid,
##                                                 ref_clf = ref_clf,
#                                                 ref_clf = None,
##                                                 strat_kfold = False,
#                                                 strat_kfold = True,
#                                                 verbose = False)
