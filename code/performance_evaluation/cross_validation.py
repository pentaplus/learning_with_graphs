import inspect
import numpy as np
import sys
import time

from collections import defaultdict
from os.path import abspath, dirname, join
from sklearn import cross_validation
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


def construct_outer_cv_and_inner_cvs(class_lbls, strat_kfold, num_outer_folds,
                                     num_inner_folds):
    if strat_kfold:
        outer_cv = cross_validation.StratifiedKFold(class_lbls,
                                                    num_outer_folds,
                                                    shuffle = True)
    else:
        outer_cv = cross_validation.KFold(len(class_lbls), num_outer_folds,
                                          shuffle = True)
    inner_cvs = []
    for outer_train_indices, outer_test_indices in outer_cv:
        if strat_kfold:
            inner_cv =\
            cross_validation.StratifiedKFold(class_lbls[outer_train_indices],
                                             num_inner_folds, shuffle = True)                                 
        else:
            inner_cv = num_inner_folds
        inner_cvs.append(inner_cv)
    
    return outer_cv, inner_cvs


def optimize_embedding_param(clf, data_mat_of_param, class_lbls, strat_kfold,
                             num_iter, num_outer_folds, num_inner_folds,
                             result_file):
    cross_val_start_time = time.time()

    mean_scores_on_test_data = []
    for i in xrange(num_iter):
        scores_on_test_data = []
        outer_cv, inner_cvs = construct_outer_cv_and_inner_cvs(class_lbls,
                                                               strat_kfold,
                                                               num_outer_folds,
                                                               num_inner_folds)
                                                               
        for j, (train_indices, test_indices) in enumerate(outer_cv):
            best_param_on_train_data = -1
            best_score_on_train_data = 0.0
            
            inner_cv = inner_cvs[j]
            
            for param, data_mat in data_mat_of_param.iteritems():
                score_on_train_data =\
                    cross_validation.cross_val_score(clf, data_mat[train_indices],
                                                     class_lbls[train_indices],
                                                     cv = inner_cv).mean()
            
                sys.stdout.write('param = %d, i = %d, j = %d: score = %.2f\n' %\
                                               (param, i, j, score_on_train_data))
                                                             
                if score_on_train_data > best_score_on_train_data:
                    best_score_on_train_data = score_on_train_data
                    best_param_on_train_data = param
             
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
        sys.stdout.write('RESULT for i = %d: %.2f\n' %\
                                                     (i, mean_score_on_test_data))
        print '-------------------------------------------------------------\n'                                             
                                                        
    cross_val_end_time = time.time()
    cross_val_time = cross_val_end_time - cross_val_start_time
    
         
    if strat_kfold:
        result_file.write('Strat: ')
    else:
        result_file.write('KFold: ')
    print '-------------------------------------------------------------'
    sys.stdout.write('TOTAL RESULT: ')
    utils.write('%.3f (+/-%.3f) in %.1f seconds\n' %\
                (np.mean(mean_scores_on_test_data),
                 np.std(mean_scores_on_test_data), cross_val_time), result_file)
    print '-------------------------------------------------------------\n'


# --------------------------------------------------------------------------------
                       
                       
#                       
#                       
#    scores = {}
#    for param, data_mat in data_mat_of_param.iteritems(): 
#        print 'evaluating (param = %d)' % param
#        
#        for i in xrange(num_iter):
#            if not i in scores:
#                scores[i] = {}
#            outer_cv, inner_cvs = cvs[i]
#            
#            for j, (train_indices, test_indices) in enumerate(outer_cv):
#                if j not in scores[i]:
#                    scores[i][j] = {'best_param_on_train_data' : -1, 'best_score' : 0.0}
#                
#                inner_cv = inner_cvs[j]
#                
#                score =\
#                cross_validation.cross_val_score(clf, data_mat[train_indices],
#                                                 class_lbls[train_indices],
#                                                 cv = inner_cv).mean()
#            
#                sys.stdout.write('param = %d, i = %d, j = %d: score = %.2f\n' %\
#                                                             (param, i, j, score))            
#            
#                if score > scores[i][j]['best_score']:
#                    scores[i][j]['best_score'] = score
#                    scores[i][j]['best_param_on_train_data'] = param
#            
#            print ''                                          
#    
#    best_param_on_train_data_values = {}
#    for i in xrange(num_iter):
#        for j in scores[i].iterkeys():
#            best_param_on_train_data = scores[i][j]['best_param_on_train_data']
#            if best_param_on_train_data not in best_param_on_train_data_values:
#                best_param_on_train_data_values[best_param_on_train_data] = {i:[j]}
#            else:
#                if i not in best_param_on_train_data_values[best_param_on_train_data].iterkeys():
#                    best_param_on_train_data_values[best_param_on_train_data][i] = [j]
#                else:
#                    best_param_on_train_data_values[best_param_on_train_data][i].append(j)
#    
#    scores = {}
#    outer_cv_lists = {}
#    for i in xrange(num_iter):    
#        scores[i] = []
#        outer_cv, inner_cvs = cvs[i]
#        outer_cv_lists[i] = list(outer_cv)
#        
#    for best_param_on_train_data in best_param_on_train_data_values.iterkeys():
#        print '\nextracting features (param = %d)' % best_param_on_train_data
#        data_mat, class_lbls = embedding.extract_features(graph_of_num,
#                                                             best_param_on_train_data)
#        for i in best_param_on_train_data_values[best_param_on_train_data].iterkeys():
#            for j in best_param_on_train_data_values[best_param_on_train_data][i]:
#                train_indices, test_indices = outer_cv_lists[i][j]
#                clf.fit(data_mat[train_indices], class_lbls[train_indices])
#                score = clf.score(data_mat[test_indices],
#                                  class_lbls[test_indices])
#                scores[i].append(score)
#                print ("i = %d, j = %d: score on test data = %.2f (for param "
#                       "= %d)") % (i, j, score, best_param_on_train_data)
#    
#    mean_scores = []            
#    for i in xrange(num_iter):
#        mean_score = np.mean(scores[i])
#        mean_scores.append(mean_score)
#        sys.stdout.write('\nRESULT for i = %d: %.2f' % (i, mean_score))
#        
#    cross_val_end_time = time.time()
#    cross_val_time = cross_val_end_time - cross_val_start_time
#    
#         
#    if strat_kfold:
#        result_file.write('Strat: ')
#    else:
#        result_file.write('KFold: ')
#    sys.stdout.write('\n\nTOTAL RESULT: ')
#    utils.write('%.3f (+/-%.3f) in %.1f seconds\n' %\
#                (np.mean(mean_scores), np.std(mean_scores), cross_val_time),
#                 result_file)
#    print '\n'
    

def optimize_embedding_and_kernel_param(graph_of_num, embedding, param_range,
                                        strat_kfold, num_iter, num_outer_folds,
                                        num_inner_folds, result_file):
    cross_val_start_time = time.time()
    
    lin_clf = svm.SVC(kernel = 'linear')
    rbf_clf = svm.SVC(kernel = 'rbf')
                                              
    data_mat, class_lbls = embedding.extract_features(graph_of_num,
                                                         min(param_range))
    # precompute KFold/StratifiedKFold objects
    cvs = {}
    for it in xrange(num_iter):
        if strat_kfold:
            outer_cv = cross_validation.StratifiedKFold(class_lbls,
                                                        num_outer_folds,
                                                        shuffle = True)
        else:
            outer_cv = cross_validation.KFold(len(class_lbls), num_outer_folds,
                                              shuffle = True)
        inner_cvs = []
        for outer_train_indices, outer_test_indices in outer_cv:
            if strat_kfold:
                inner_cv =\
                cross_validation.StratifiedKFold(class_lbls[outer_train_indices],
                                                 num_inner_folds, shuffle = True)                                 
            else:
                inner_cv = num_inner_folds
            inner_cvs.append(inner_cv)
            
        cvs[it] = (outer_cv, inner_cvs)
                                    
    scores = {}
    for param in param_range:  
        print 'extracting features (param = %d)' % param
        data_mat, class_lbls = embedding.extract_features(graph_of_num, param)
        
        for i in xrange(num_iter):
            if not i in scores:
                scores[i] = {}
            outer_cv, inner_cvs = cvs[i]
            
            for j, (train_indices, test_indices) in enumerate(outer_cv):
                if j not in scores[i]:
                    scores[i][j] = {'best_param_on_train_data' : -1, 'best_score' : 0.0}
                
                inner_cv = inner_cvs[j]
                
                score =\
                cross_validation.cross_val_score(lin_clf,
                                                 data_mat[train_indices],
                                                 class_lbls[train_indices],
                                                 cv = inner_cv).mean()
            
                sys.stdout.write('param = %d, i = %d, j = %d: score = %.2f\n' %\
                                                             (param, i, j, score))            
            
                if score > scores[i][j]['best_score']:
                    scores[i][j]['best_score'] = score
                    scores[i][j]['best_param_on_train_data'] = param
            
            print ''                                          
    
    best_param_on_train_data_values = {}
    for i in xrange(num_iter):
        for j in scores[i].iterkeys():
            best_param_on_train_data = scores[i][j]['best_param_on_train_data']
            if best_param_on_train_data not in best_param_on_train_data_values:
                best_param_on_train_data_values[best_param_on_train_data] = {i:[j]}
            else:
                if i not in best_param_on_train_data_values[best_param_on_train_data].iterkeys():
                    best_param_on_train_data_values[best_param_on_train_data][i] = [j]
                else:
                    best_param_on_train_data_values[best_param_on_train_data][i].append(j)
    
    scores = {}
    outer_cv_lists = {}
    for i in xrange(num_iter):    
        scores[i] = []
        outer_cv, inner_cvs = cvs[i]
        outer_cv_lists[i] = list(outer_cv)
        
    for best_param_on_train_data in best_param_on_train_data_values.iterkeys():
        print '\nextracting features (param = %d)' % best_param_on_train_data
        data_mat, class_lbls = embedding.extract_features(graph_of_num,
                                                             best_param_on_train_data)
        for i in best_param_on_train_data_values[best_param_on_train_data].iterkeys():
            for j in best_param_on_train_data_values[best_param_on_train_data][i]:
                train_indices, test_indices = outer_cv_lists[i][j]
                                          
                lin_train_score =\
                cross_validation.cross_val_score(lin_clf,
                                                 data_mat[train_indices],
                                                 class_lbls[train_indices],
                                                 cv = 3).mean()
                                                 
                rbf_train_score =\
                cross_validation.cross_val_score(rbf_clf,
                                                 data_mat[train_indices],
                                                 class_lbls[train_indices],
                                                 cv = 3).mean()
                                                 
                if lin_train_score > rbf_train_score:
                    lin_clf.fit(data_mat[train_indices], 
                                class_lbls[train_indices])
                    score = lin_clf.score(data_mat[test_indices],
                                          class_lbls[test_indices])
                else:
                    rbf_clf.fit(data_mat[train_indices], 
                                class_lbls[train_indices])
                    score = rbf_clf.score(data_mat[test_indices],
                                          class_lbls[test_indices])
                    
                scores[i].append(score)
                
                print ("i = %d, j = %d: score on test data = %.2f (for param = "
                       "%d)") % (i, j, score, best_param_on_train_data)
    
    mean_scores = []            
    for i in xrange(num_iter):
        mean_score = np.mean(scores[i])
        mean_scores.append(mean_score)
        sys.stdout.write('\nRESULT for i = %d: %.2f' % (i, mean_score))
        
    cross_val_end_time = time.time()
    cross_val_time = cross_val_end_time - cross_val_start_time
    
         
    if strat_kfold:
        result_file.write('Strat: ')
    else:
        result_file.write('KFold: ')
    sys.stdout.write('\n\nTOTAL RESULT: ')
    utils.write('%.3f (+/-%.3f) in %.1f seconds\n' %\
                (np.mean(mean_scores), np.std(mean_scores), cross_val_time),
                 result_file)
    print '\n'


def cross_val(clf, data_mat, class_lbls, num_iter, num_folds, strat_kfold,
              result_file):
    cross_val_start_time = time.time()

    mean_scores = []
    for i in xrange(num_iter):
        if strat_kfold:
            cv = cross_validation.StratifiedKFold(class_lbls, num_folds,
                                                  shuffle = True)
        else:
            cv = cross_validation.KFold(len(class_lbls), num_folds,
                                        shuffle = True)
            
        scores = cross_validation.cross_val_score(clf, data_mat, class_lbls,
                                                  cv = cv)
        
        print '%d) score: %.2f' % (i, scores.mean())
        
        mean_scores.append(scores.mean())    
    print ''
      
    cross_val_end_time = time.time()
    cross_val_time = cross_val_end_time - cross_val_start_time
          
    if strat_kfold:
        result_file.write('Strat: ')
    else:
        result_file.write('KFold: ')
    sys.stdout.write('RESULT: ')
    utils.write('%.3f (+/-%.3f) in %.1f seconds\n' %\
                (np.mean(mean_scores), np.std(mean_scores), cross_val_time),
                 result_file)
    sys.stdout.write('\n')
    
    
def optimize_gen_params(data_mat, class_lbls, num_iter, param_grid, num_folds,
                        strat_kfold, verbose, result_file):
    cross_val_start_time = time.time()
    
    grid_clf = GridSearchCV(svm.SVC(), param_grid, cv = 3)
    
    mean_scores = []            
    
    for i in xrange(num_iter):
        if strat_kfold:
            cv = cross_validation.StratifiedKFold(class_lbls, num_folds,
                                                  shuffle = True)
        else:
            cv = num_folds
        
        scores = []
        for j, (train_indices, test_indices) in enumerate(cv):
            # print("TRAIN:", train_indices, "TEST:", test_indices)
        
            grid_clf.fit(data_mat[train_indices], class_lbls[train_indices])
            
            #for params, mean_score, scores in clf.grid_scores_:
            #    print params, mean_score
            #print clf.best_param_on_train_datas_
            
            opt_clf = grid_clf.best_estimator_
            score = opt_clf.score(data_mat[test_indices],
                                  class_lbls[test_indices])
            scores.append(score)
            
            if verbose:
                print '%d.%d) score: %.2f' % (i, j, score)
        
        print '%d) score: %.2f' % (i, np.mean(scores))
        
        if verbose:
            print ''

        mean_scores.append(np.mean(scores)) 

    if not verbose:
        print ''
                                            
    cross_val_end_time = time.time()
    cross_val_time = cross_val_end_time - cross_val_start_time
    
    if param_grid == [{'kernel': ['linear', 'rbf']}]:
        if not strat_kfold:
            result_file.write('OPT (LINEAR/RBF)\n')
    if strat_kfold:
        result_file.write('Strat: ')
    else:
        result_file.write('KFold: ')
    sys.stdout.write('RESULT: ')
    utils.write('%.3f (+/-%.3f) in %.1f seconds\n' %\
                (np.mean(mean_scores), np.std(mean_scores), cross_val_time),
                result_file)
    print '\n'
    if strat_kfold:
    	result_file.write('\n') 
	

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
    