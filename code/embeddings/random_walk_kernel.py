from __future__ import division
import inspect
import networkx as nx
import numpy as np
import sys
import time

from control import dlyap
from os.path import abspath, dirname, join
from scipy.sparse.linalg import cg, LinearOperator


# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from misc import utils, pcg, pz


def vec(M):
    return M.reshape((M.shape[0] * M.shape[1], 1))
    

def invvec(M, m, n):
    return M.reshape((m, n))
    
    
def smtfilter(x, A_i, A_j, lmbd):
    yy = vec(A_i.dot(invvec(x, A_i.shape[0], A_j.shape[0])).dot(A_j))
    
    yy *= lmbd
    
    vecu = x - yy
    
    return vecu
    

def compute_kernel_mat(graph_meta_data_of_num, param_range = [None]):
    kernel_mat_comp_start_time = time.time()
    
    kernel_mat_comp_time_of_param = {}
    kernel_mat_of_param = {}    
    
    
    num_graphs = len(graph_meta_data_of_num)
    graph_meta_data = graph_meta_data_of_num.values()
    
    kernel_mat = np.zeros((num_graphs, num_graphs), dtype = np.float64)
    
    lmbd = -2
    
        
    
    # iterate over all graphs in the dataset -------------------------------------
    for i in xrange(num_graphs):
        # load graph       
        G_i = pz.load(graph_meta_data[i][0])
        # determine adjacency matrix A of graph G
#        A_i = utils.get_adjacency_matrix(G_i)
        A_i = nx.adjacency_matrix(G_i, weight = None).todense()

        for j in xrange(i, num_graphs):
            # load graph       
            G_j = pz.load(graph_meta_data[j][0])
            # determine adjacency matrix A of graph G
#            A_j = utils.get_adjacency_matrix(G_j)
            A_j = nx.adjacency_matrix(G_j, weight = None).todense()
            
            smtfiler_op = LinearOperator((A_i.shape[0] * A_j.shape[0],
                                          A_i.shape[0] * A_j.shape[0]),
                                         lambda x: smtfilter(x, A_i, A_j, lmbd))            
            
#            C = np.ones((A_j.shape[0], A_i.shape[0]))
            
            b = np.ones((A_i.shape[0] * A_j.shape[0], 1))
            
            x, info = cg(smtfiler_op, b,
                         x0 = np.zeros((A_i.shape[0] * A_j.shape[0])), tol = 1e-6,
                         maxiter = 20)
            
            x, info = cg(smtfiler_op, b, tol = 1e-6, maxiter = 20)
            
#            x, flag, relres, iter_, resvec = pcg.pcg(lambda x: smtfilter(x, A_i, A_j, lmbd), b, 1e-6, 20)
            
#            return X, info
#            
            kernel_mat[i,j] = np.sum(x)
#            
#            # i = 0, j = 0: 38.926
            print 'i =', i, 'j =', j, kernel_mat[i,j]
#            
#            Y = A_j.dot(X).dot((lmbd * A_i).T) - X + C
#            
#            x = 0
        
        
    

    kernel_mat_of_param[None] = kernel_mat
    
    kernel_mat_comp_end_time = time.time()
    kernel_mat_comp_time_of_param[None] =\
                             kernel_mat_comp_end_time - kernel_mat_comp_start_time

    return kernel_mat_of_param, kernel_mat_comp_time_of_param


if __name__ == '__main__':
    from misc import dataset_loader, utils
    
    DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', 'datasets')
    dataset = 'MUTAG'
#    dataset = 'DD'
#    dataset = 'ENZYMES'
#    dataset = 'NCI1'
#    dataset = 'NCI109'
    graph_meta_data_of_num, class_lbls =\
      dataset_loader.get_graph_meta_data_of_num_dict_and_class_lbls(dataset,
                                                                    DATASETS_PATH)
    
    
    kernel_mat_of_param, kernel_mat_comp_time_of_param =\
                  compute_kernel_mat(graph_meta_data_of_num, param_range = [None])

    X, info = compute_kernel_mat(graph_meta_data_of_num, param_range = [None])
    
    kernel_mat = kernel_mat_of_param[None]
    kernel_mat_comp_time = kernel_mat_comp_time_of_param[None]
    print kernel_mat_comp_time
    
    
    import networkx as nx
    from scipy.sparse import csr_matrix
    import scipy.io as spio
    G = pz.load(graph_meta_data_of_num.values()[0][0])
    A = nx.adjacency_matrix(G, weight = None).todense()
    
    A_sprs = csr_matrix(A)
    A_sprs
    I = np.nonzero(A)
    I[0]
    
    mat = spio.loadmat('data.mat')
    
    A_mat = mat['A']
    
    
    lmbd = -2
    G = pz.load(graph_meta_data_of_num.values()[0][0])
#    A = nx.adjacency_matrix(G, weight = None).todense()
    A = utils.get_adjacency_matrix(G)
    
#    smtfiler_op = LinearOperator((A.shape[0] * A.shape[0],
#                                  A.shape[0] * A.shape[0]),
#                                 lambda x: smtfilter(x, A, A, lmbd))
                                 
    b = np.ones((A.shape[0] * A.shape[0], 1))
#    x, info = cg(smtfiler_op, b, tol = 1e-6, maxiter = 20)
    
#    x = np.zeros((529,1))
#    out = smtfilter(x, A, A, lmbd)
    
        
    
    f = lambda x: smtfilter(x, A, A, lmbd)
    
    x = np.zeros((529, 1))
    
    f(x)
   