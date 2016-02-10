import inspect
import sys

from os.path import abspath, dirname, join


# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from embeddings import neighborhood_hash_main


def extract_features(graph_of_num, h):
    return neighborhood_hash_main.extract_features(graph_of_num, h,
                                                   count_sensitive = False,
                                                   all_iter = True)
    

# !!
if __name__ == '__main__':
    import time
    from misc import datasetloader
    
    DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', 'datasets')
    dataset = 'MUTAG'
    graph_of_num = datasetloader.load_dataset(DATASETS_PATH, dataset)
    
    
    h = 9
    start = time.time()
    data_matrix, class_lbls = extract_features(graph_of_num, h)
    
    end = time.time()
    print 'h = %d: %.3f' % (h, end - start)
    
    
#    Z = data_matrix.todense()
    
    print data_matrix.__repr__()
    #print data_matrix.__str__()
        