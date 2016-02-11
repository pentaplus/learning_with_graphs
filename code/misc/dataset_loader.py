import inspect
import pz
import re
import sys

from collections import OrderedDict
from os import listdir
from os.path import abspath, dirname, join

# determine script path
FILE_NAME = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_PATH = dirname(abspath(FILE_NAME))

from misc import utils


def determine_folder_of_dataset_dict(datasets_path):
    folder_of_dataset = {}
    folders = utils.list_sub_dirs(datasets_path)
    for folder in folders:
        m = re.match('.*(?= \()', folder)
        if not m:
            continue
        
        dataset_name = m.group(0)
        folder_of_dataset[dataset_name] = folder
        
    return folder_of_dataset
    
    
def determine_folder_of_class_dict(classes_path):
    folder_of_class = {}
    dataset_classes = listdir(classes_path)
    for dataset_class in dataset_classes:
        reg_exp = '(?<=class )-?\d+(?= \()' if '(' in dataset_class\
                                                           else '(?<=class )-?\d+'
        m = re.search(reg_exp, dataset_class)
        if not m:
            continue
        
        class_lbl = int(m.group(0))
        folder_of_class[class_lbl] = dataset_class
    
    return folder_of_class
    
    
def determine_graph_of_num_dict(classes_path, folder_of_class):
    # The values of graph_of_num are pairs of the form
    # (nx.Graph/nx.DiGraph, class_lbl)
    graph_of_num = {}
    
    for class_lbl, folder in folder_of_class.iteritems():
        path_to_graphs_of_cur_class = join(classes_path, folder)    
        
        for graph_file in utils.list_files(path_to_graphs_of_cur_class):
            m = re.match('\d+(?=.pz)', graph_file)
            if not m:
                continue
            
            graph_num = int(m.group(0))
            # load graph with number graph_num
            cur_graph = pz.load(join(path_to_graphs_of_cur_class, graph_file))
            graph_of_num[graph_num] = (cur_graph, class_lbl)
            
    return OrderedDict(sorted(graph_of_num.iteritems()))
    
    
def determine_graphs_of_class_dict(graph_of_num):
    graphs_of_class = {}

    for graph_num, (graph, class_lbl) in graph_of_num.iteritems():
        if class_lbl not in graphs_of_class:
            graphs_of_class[class_lbl] = [(graph_num, graph)]
        else:
            graphs_of_class[class_lbl].append((graph_num, graph))
        
    return graphs_of_class
    
    
def get_class_lbls(graph_of_num):
    class_lbls = []
    for graph, class_lbl in graph_of_num.itervalues():
        class_lbls.append(class_lbl)
        
    return class_lbls
    
    
def load_dataset(datasets_path, dataset):
    folder_of_dataset = determine_folder_of_dataset_dict(datasets_path)
    
    datasets = folder_of_dataset.keys()
    if not dataset in datasets:
        print '%s is not a valid dataset name.' % dataset
        sys.exit(1)
    
    classes_path = join(datasets_path, folder_of_dataset[dataset], 'pz')

    folder_of_class = determine_folder_of_class_dict(classes_path)

    return determine_graph_of_num_dict(classes_path, folder_of_class)