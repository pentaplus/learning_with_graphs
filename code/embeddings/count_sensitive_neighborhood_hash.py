import inspect
import sys

from os.path import abspath, dirname, join


# determine script path
filename = inspect.getframeinfo(inspect.currentframe()).filename
script_path = dirname(abspath(filename))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(script_path, '..'))

from embeddings import neighborhood_hash_main


def extract_features(graph_of_num, h):
    return neighborhood_hash_main.extract_features(graph_of_num, h)
    

# !!
if __name__ == '__main__':
    import time
    from misc import datasetloader
    
    DATASETS_PATH = join(script_path, '..', '..', 'datasets')
    dataset = 'MUTAG'
    graph_of_num = datasetloader.load_dataset(DATASETS_PATH, dataset)
    
    
    h = 9
    start = time.time()
    data_matrix, class_lbls = extract_features(graph_of_num, h)
    
    end = time.time()
    print 'h = %d: %.3f' % (h, end - start)
    
    
    Z = data_matrix.todense()
    
    print data_matrix.__repr__()
    #print data_matrix.__str__()
        