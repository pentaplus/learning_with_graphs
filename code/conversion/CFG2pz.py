import inspect
import networkx as nx
import os
import re
import sys

from os.path import abspath, dirname, join, splitext


# determine script path
FILE_NAME = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_PATH = dirname(abspath(FILE_NAME))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_PATH, '..'))

from misc import dataset_loader, pz, utils


def determine_graph_of_num_dict(source_classes_path, folder_of_class):
    # The values of graph_of_num are pairs of the form
    # (nx.Graph/nx.DiGraph, class_lbl)
    graph_of_num = {}
    
    for class_lbl, folder in folder_of_class.iteritems():
        path_to_graphs_of_cur_class = join(source_classes_path, folder)    
        
        for graph_file in utils.list_files(path_to_graphs_of_cur_class):
            m = re.match('\d+(?=.pz)', graph_file)
            if not m:
                continue
            
            graph_num = int(m.group(0))
            # load graph with number graph_num
            cur_graph = pz.load(join(path_to_graphs_of_cur_class, graph_file))
            graph_of_num[graph_num] = (cur_graph, class_lbl)
            
#    return OrderedDict(sorted(graph_of_num.iteritems()))


SOURCE_CLASSES_PATH = join(SCRIPT_PATH, '..', '..', 'datasets', ('CFG (2 '
                           'classes, 3193 directed graphs, unlabeled edges)'),
                           'plain')
                    
folder_of_class =\
                dataset_loader.determine_folder_of_class_dict(SOURCE_CLASSES_PATH)

determine_graph_of_num_dict(SOURCE_CLASSES_PATH, folder_of_class)

utils.check_for_pz_folder()


for class_lbl, folder in folder_of_class.iteritems():
    source_class_path = join(SOURCE_CLASSES_PATH, folder)
    target_class_path = join('pz', folder)
    os.makedirs(target_class_path)
    
    for file_name in utils.list_files(source_class_path):
        base_file_name, file_extension = splitext(file_name)
        
        if not file_extension == '.cfg':
            continue
        
        with open(join(source_class_path, file_name)) as f:
            # --------------------------------------------------------------------
            # 1) parse graph
            # --------------------------------------------------------------------
        
        
            # --------------------------------------------------------------------
            # 2) create a networkx graph corresponding to the parsed graph
            # --------------------------------------------------------------------
            G = nx.DiGraph()
            
            pz.save(G, join(target_class_path, base_file_name + '.pz'))
            
        
        sys.exit(0)