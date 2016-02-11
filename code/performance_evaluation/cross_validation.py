import inspect
import numpy as np
import sys
import time

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

from misc import utils, dataset_loader


def optimize_embedding_param(clf, graph_of_num, embedding, param_range,
                             strat_kfold, num_iter, num_outer_folds,
                             num_inner_folds, result_file):
    cross_val_start_time = time.time()                                              
    
    class_lbls = np.array(dataset_loader.get_class_lbls(graph_of_num))

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
        data_matrix, class_lbls = embedding.extract_features(graph_of_num, param)
        
        for i in xrange(num_iter):
            if not i in scores:
                scores[i] = {}
            outer_cv, inner_cvs = cvs[i]
            
            for j, (train_indices, test_indices) in enumerate(outer_cv):
                if j not in scores[i]:
                    scores[i][j] = {'best_param' : -1, 'best_score' : 0.0}
                
                inner_cv = inner_cvs[j]
                
                score =\
                cross_validation.cross_val_score(clf, data_matrix[train_indices],
                                                 class_lbls[train_indices],
                                                 cv = inner_cv).mean()
            
                sys.stdout.write('param = %d, i = %d, j = %d: score = %.2f\n' %\
                                                             (param, i, j, score))            
            
                if score > scores[i][j]['best_score']:
                    scores[i][j]['best_score'] = score
                    scores[i][j]['best_param'] = param
            
            print ''                                          
    
    best_param_values = {}
    for i in xrange(num_iter):
        for j in scores[i].iterkeys():
            best_param = scores[i][j]['best_param']
            if best_param not in best_param_values:
                best_param_values[best_param] = {i:[j]}
            else:
                if i not in best_param_values[best_param].iterkeys():
                    best_param_values[best_param][i] = [j]
                else:
                    best_param_values[best_param][i].append(j)
    
    scores = {}
    outer_cv_lists = {}
    for i in xrange(num_iter):    
        scores[i] = []
        outer_cv, inner_cvs = cvs[i]
        outer_cv_lists[i] = list(outer_cv)
        
    for best_param in best_param_values.iterkeys():
        print '\nextracting features (param = %d)' % best_param
        data_matrix, class_lbls = embedding.extract_features(graph_of_num,
                                                             best_param)
        for i in best_param_values[best_param].iterkeys():
            for j in best_param_values[best_param][i]:
                train_indices, test_indices = outer_cv_lists[i][j]
                clf.fit(data_matrix[train_indices], class_lbls[train_indices])
                score = clf.score(data_matrix[test_indices],
                                  class_lbls[test_indices])
                scores[i].append(score)
                print ("i = %d, j = %d: score on test data = %.2f (for param "
                       "= %d)") % (i, j, score, best_param)
    
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
    

def optimize_embedding_and_kernel_param(graph_of_num, embedding, param_range,
                                        strat_kfold, num_iter, num_outer_folds,
                                        num_inner_folds, result_file):
    cross_val_start_time = time.time()
    
    lin_clf = svm.SVC(kernel = 'linear')
    rbf_clf = svm.SVC(kernel = 'rbf')
                                              
    data_matrix, class_lbls = embedding.extract_features(graph_of_num,
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
        data_matrix, class_lbls = embedding.extract_features(graph_of_num, param)
        
        for i in xrange(num_iter):
            if not i in scores:
                scores[i] = {}
            outer_cv, inner_cvs = cvs[i]
            
            for j, (train_indices, test_indices) in enumerate(outer_cv):
                if j not in scores[i]:
                    scores[i][j] = {'best_param' : -1, 'best_score' : 0.0}
                
                inner_cv = inner_cvs[j]
                
                score =\
                cross_validation.cross_val_score(lin_clf,
                                                 data_matrix[train_indices],
                                                 class_lbls[train_indices],
                                                 cv = inner_cv).mean()
            
                sys.stdout.write('param = %d, i = %d, j = %d: score = %.2f\n' %\
                                                             (param, i, j, score))            
            
                if score > scores[i][j]['best_score']:
                    scores[i][j]['best_score'] = score
                    scores[i][j]['best_param'] = param
            
            print ''                                          
    
    best_param_values = {}
    for i in xrange(num_iter):
        for j in scores[i].iterkeys():
            best_param = scores[i][j]['best_param']
            if best_param not in best_param_values:
                best_param_values[best_param] = {i:[j]}
            else:
                if i not in best_param_values[best_param].iterkeys():
                    best_param_values[best_param][i] = [j]
                else:
                    best_param_values[best_param][i].append(j)
    
    scores = {}
    outer_cv_lists = {}
    for i in xrange(num_iter):    
        scores[i] = []
        outer_cv, inner_cvs = cvs[i]
        outer_cv_lists[i] = list(outer_cv)
        
    for best_param in best_param_values.iterkeys():
        print '\nextracting features (param = %d)' % best_param
        data_matrix, class_lbls = embedding.extract_features(graph_of_num,
                                                             best_param)
        for i in best_param_values[best_param].iterkeys():
            for j in best_param_values[best_param][i]:
                train_indices, test_indices = outer_cv_lists[i][j]
                                          
                lin_train_score =\
                cross_validation.cross_val_score(lin_clf,
                                                 data_matrix[train_indices],
                                                 class_lbls[train_indices],
                                                 cv = 3).mean()
                                                 
                rbf_train_score =\
                cross_validation.cross_val_score(rbf_clf,
                                                 data_matrix[train_indices],
                                                 class_lbls[train_indices],
                                                 cv = 3).mean()
                                                 
                if lin_train_score > rbf_train_score:
                    lin_clf.fit(data_matrix[train_indices], 
                                class_lbls[train_indices])
                    score = lin_clf.score(data_matrix[test_indices],
                                          class_lbls[test_indices])
                else:
                    rbf_clf.fit(data_matrix[train_indices], 
                                class_lbls[train_indices])
                    score = rbf_clf.score(data_matrix[test_indices],
                                          class_lbls[test_indices])
                    
                scores[i].append(score)
                
                print ("i = %d, j = %d: score on test data = %.2f (for param = "
                       "%d)") % (i, j, score, best_param)
    
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


def cross_val(clf, data_matrix, class_lbls, num_iter, num_folds, strat_kfold,
              result_file):
    cross_val_start_time = time.time()

    mean_scores = []
    for i in xrange(num_iter):
        if strat_kfold:
            cv = cross_validation.StratifiedKFold(class_lbls, num_folds,
                                                  shuffle = True)
        else:
#            cv = num_folds
            cv = cross_validation.KFold(len(class_lbls), num_folds,
                                        shuffle = True)
            
        scores = cross_validation.cross_val_score(clf, data_matrix, class_lbls,
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
    print '\n'
    
    
def optimize_gen_params(data_matrix, class_lbls, num_iter, param_grid, num_folds,
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
        
            grid_clf.fit(data_matrix[train_indices], class_lbls[train_indices])
            
            #for params, mean_score, scores in clf.grid_scores_:
            #    print params, mean_score
            #print clf.best_params_
            
            opt_clf = grid_clf.best_estimator_
            score = opt_clf.score(data_matrix[test_indices],
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
def loo_cross_val(clf, data_matrix, class_lbls):
    """leave-one-out cross-validation"""
    N = data_matrix.shape[0]
    matches = 0
    
    for i in xrange(N):
        train_indices = range(N)
        train_indices.remove(i)
        
        test_index = i
        
        clf.fit(data_matrix[train_indices],
                [class_lbls[i] for i in train_indices])
        
        
        print(data_matrix[test_index].todense().__str__(),
              class_lbls[test_index],
              int(clf.predict(data_matrix[test_index])))
              
        if int(clf.predict(data_matrix[test_index])) == class_lbls[test_index]:
            matches += 1
    
    print ''
    print 'avg score: %.2f' % (float(matches)/N)
	

	
# !!
def compute_kernel_matrix(data_matrix):
    K = pairwise_kernels(data_matrix)
    
    entries = []
    for i in xrange(data_matrix.shape[0]):
        x = np.asarray(data_matrix[i].todense())
        entries.append((K[i,i], np.vdot(x, x)))
        
    entries_array = np.array(entries)
    (entries_array[:,0] - entries_array[:,1]).max()
    