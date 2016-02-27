import inspect
import networkx as nx
import numpy as np
import sys
import time

from os.path import abspath, dirname, join


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
    
    
def smtfilter(x, A_i, A_j, lambda_):
    yy = vec(A_i.dot(invvec(x, A_i.shape[0], A_j.shape[0])).dot(A_j))
    
    yy *= lambda_
    
    vecu = x - yy
    
    return vecu
    

def compute_kernel_mat(graph_meta_data_of_num, param_range = [None]):
    kernel_mat_comp_start_time = time.time()
    
    kernel_mat_comp_time_of_param = {}
    kernel_mat_of_param = {}    
    
    
    num_graphs = len(graph_meta_data_of_num)
    graph_meta_data = graph_meta_data_of_num.values()
    
    kernel_mat = np.zeros((num_graphs, num_graphs), dtype = np.float64)
    
    lambda_ = -2
        
    
    # iterate over all graphs in the dataset -------------------------------------
    for i in xrange(num_graphs):
        # load graph       
        G_i = pz.load(graph_meta_data[i][0])
        # determine adjacency matrix A of graph G
        A_i = utils.get_adjacency_matrix(G_i)

        for j in xrange(i, num_graphs):
            # load graph       
            G_j = pz.load(graph_meta_data[j][0])
            # determine adjacency matrix A of graph G
            A_j = utils.get_adjacency_matrix(G_j)
            
            # apply preconditioned conjugate gradient method (the pcg.pcg
            # function is a translation of the MATLAB pcg to Python,
            # see http://de.mathworks.com/help/matlab/ref/pcg.html for further
            # details)
            b = np.ones((A_i.shape[0] * A_j.shape[0], 1))
            
            x, flag, relres, iter_, resvec =\
                   pcg.pcg(lambda x: smtfilter(x, A_i, A_j, lambda_), b, 1e-6, 20)
                
            
            kernel_mat[i,j] = np.sum(x)
            if i != j:
                kernel_mat[j,i] = kernel_mat[i,j]
            
#             # !!
##            sys.modules['__main__'].kernel_mat = kernel_mat
            
#            print 'i =', i, 'j =', j
            print 'i =', i, 'j =', j, kernel_mat[i,j]

        
        
    

    kernel_mat_of_param[None] = kernel_mat
    
    kernel_mat_comp_end_time = time.time()
    kernel_mat_comp_time_of_param[None] =\
                             kernel_mat_comp_end_time - kernel_mat_comp_start_time

    return kernel_mat_of_param, kernel_mat_comp_time_of_param


if __name__ == '__main__':
    from misc import dataset_loader, utils
    
    DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', 'datasets')
#    dataset = 'MUTAG'
#    dataset = 'PTC(MR)'
    dataset = 'DD'
#    dataset = 'ENZYMES'
#    dataset = 'NCI1'
#    dataset = 'NCI109'
    graph_meta_data_of_num, class_lbls =\
      dataset_loader.get_graph_meta_data_of_num_dict_and_class_lbls(dataset,
                                                                    DATASETS_PATH)
    
    
    kernel_mat_of_param, kernel_mat_comp_time_of_param =\
                  compute_kernel_mat(graph_meta_data_of_num, param_range = [None])

    
    kernel_mat = kernel_mat_of_param[None]
    kernel_mat_comp_time = kernel_mat_comp_time_of_param[None]
    print 'kernel_mat_comp_time = ', kernel_mat_comp_time
    
    
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
    
    # utils: 24.0, adj_mat calc: 12.7

    