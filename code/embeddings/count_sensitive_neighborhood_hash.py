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


def extract_features(graph_meta_data_of_num, h_range):
    return neighborhood_hash_main.extract_features(graph_meta_data_of_num,
                                                   h_range,
                                                   count_sensitive = True,
                                                   all_iter = False)
    

# !!
if __name__ == '__main__':
    import time
    from misc import dataset_loader
        
    DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', 'datasets')
    dataset = 'MUTAG'
    
    graph_meta_data_of_num, class_lbls =\
      dataset_loader.get_graph_meta_data_of_num_dict_and_class_lbls(dataset,
                                                                    DATASETS_PATH)
    
    h_range = range(6)
    start = time.time()
    data_mat_of_param, extr_time_of_param =\
                                 extract_features(graph_meta_data_of_num, h_range)
    
    end = time.time()
    print 'h_range = %s: %.3f' % (h_range, end - start)
    
    print '%r' % data_mat_of_param
    
    
        