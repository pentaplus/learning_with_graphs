import inspect
import numpy as np
import sys
import time

from os.path import abspath, dirname, join
from sklearn.cross_validation import cross_val_score, KFold
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics.pairwise import pairwise_kernels

# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from misc import utils


def optimize_embedding_param(clf, data_mat_of_param, class_lbls,
                             num_iter, num_outer_folds, num_inner_folds,
                             result_file):
    cross_val_start_time = time.time()

    mean_scores_on_test_data = []
    for i in xrange(num_iter):
        scores_on_test_data = []
        outer_cv = KFold(len(class_lbls), num_outer_folds, shuffle = True)
                                                               
        for j, (train_indices, test_indices) in enumerate(outer_cv):
            best_param_on_train_data = -1
            best_score_on_train_data = 0.0
            opt_clf = None

            for param, data_mat in data_mat_of_param.iteritems():
                if isinstance(clf, GridSearchCV):
                    clf.fit(data_mat[train_indices], class_lbls[train_indices])
                    
                    sub_clf = clf.best_estimator_
                    
                    print 'param = %d, i = %d, j = %d: params = %s' %\
                                                   (param, i, j, clf.best_params_)
                                                 
                    score_on_train_data = sub_clf.score(data_mat[test_indices],
                                                        class_lbls[test_indices])
                    if score_on_train_data > best_score_on_train_data:
                        opt_clf = clf
                else:
                    # clf is an instance of LinearSVC
                    score_on_train_data =\
                                     cross_val_score(clf, data_mat[train_indices],
                                                     class_lbls[train_indices],
                                                     cv = num_inner_folds).mean()
            
                print 'param = %d, i = %d, j = %d: score = %.2f' %\
                                                (param, i, j, score_on_train_data)
                                                             
                if score_on_train_data > best_score_on_train_data:
                    best_score_on_train_data = score_on_train_data
                    best_param_on_train_data = param
                    opt_clf = clf
             
            if isinstance(clf, GridSearchCV):
                clf = opt_clf
                
            best_data_mat = data_mat_of_param[best_param_on_train_data]
            clf.fit(best_data_mat[train_indices], class_lbls[train_indices])
            score_on_test_data = clf.score(best_data_mat[test_indices],
                                           class_lbls[test_indices])
            scores_on_test_data.append(score_on_test_data)
            print ('-> score on test data = %.2f (best param '
                   '= %d)\n') % (score_on_test_data, best_param_on_train_data)
       
        mean_score_on_test_data = np.mean(scores_on_test_data) 
        mean_scores_on_test_data.append(mean_score_on_test_data)

        print '-------------------------------------------------------------'
        print 'RESULT for i = %d: %.2f' % (i, mean_score_on_test_data)
        print '-------------------------------------------------------------\n'                                             
                                                        
    cross_val_end_time = time.time()
    cross_val_time = cross_val_end_time - cross_val_start_time
    
    print '-------------------------------------------------------------'
    sys.stdout.write('TOTAL RESULT: ')
    utils.write('%.3f (+/-%.3f) in %.1f seconds\n' %\
                (np.mean(mean_scores_on_test_data),
                 np.std(mean_scores_on_test_data), cross_val_time), result_file)
    print '-------------------------------------------------------------\n'


    
def cross_val(clf, data_or_kernel_mat, class_lbls, num_iter, num_outer_folds,
              result_file):
    cross_val_start_time = time.time()

    mean_scores_on_test_data = []   
    for i in xrange(num_iter):
        scores_on_test_data = []
        outer_cv = KFold(len(class_lbls), num_outer_folds, shuffle = True)
        
        if isinstance(clf, GridSearchCV):
            data_mat = data_or_kernel_mat
                                            
            for j, (train_indices, test_indices) in enumerate(outer_cv):
                clf.fit(data_mat[train_indices], class_lbls[train_indices])
                    
                opt_clf = clf.best_estimator_
                    
                print 'i = %d, j = %d: params = %s' % (i, j, clf.best_params_)
                                                 
                score_on_test_data = opt_clf.score(data_mat[test_indices],
                                                   class_lbls[test_indices])
            
                scores_on_test_data.append(score_on_test_data)
        else:
            # clf is an instance of LinearSVC or an instance of SVC with
            # precomputed kernel
            scores_on_test_data = cross_val_score(clf, data_or_kernel_mat,
                                                  class_lbls, cv = outer_cv)
                                                  
        mean_score_on_test_data = np.mean(scores_on_test_data) 
        mean_scores_on_test_data.append(mean_score_on_test_data)
        print '%d) score: %.2f' % (i, mean_score_on_test_data)  
    print ''
      
    cross_val_end_time = time.time()
    cross_val_time = cross_val_end_time - cross_val_start_time
          
    sys.stdout.write('RESULT: ')
    utils.write('%.3f (+/-%.3f) in %.1f seconds\n' %\
                               (np.mean(mean_scores_on_test_data),
                                np.std(mean_scores_on_test_data), cross_val_time),
                                result_file)
    sys.stdout.write('\n')
    
    
	

# !!
def loo_cross_val(clf, data_mat, class_lbls):
    """leave-one-out cross-validation"""
    N = data_mat.shape[0]
    matches = 0
    
    for i in xrange(N):
        train_indices = range(N)
        train_indices.remove(i)
        
        test_index = i
        
        clf.fit(data_mat[train_indices],
                [class_lbls[i] for i in train_indices])
        
        
        print(data_mat[test_index].todense().__str__(),
              class_lbls[test_index],
              int(clf.predict(data_mat[test_index])))
              
        if int(clf.predict(data_mat[test_index])) == class_lbls[test_index]:
            matches += 1
    
    print ''
    print 'avg score: %.2f' % (float(matches)/N)
	

	
# !!
def compute_kernel_matrix(data_mat):
    K = pairwise_kernels(data_mat)
    
    entries = []
    for i in xrange(data_mat.shape[0]):
        x = np.asarray(data_mat[i].todense())
        entries.append((K[i,i], np.vdot(x, x)))
        
    entries_array = np.array(entries)
    (entries_array[:,0] - entries_array[:,1]).max()
    
    
    

