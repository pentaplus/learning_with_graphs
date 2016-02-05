import pz
import re
import sys
from collections import OrderedDict
from os import listdir
from os.path import isfile, join


def list_files(path):
    return [f for f in listdir(path) if isfile(join(path, f))]


def determine_folder_of_dataset_dict(datasets_path):
    folder_of_dataset = {}
    dataset_folders = listdir(datasets_path)
    for dataset_folder in dataset_folders:
        m = re.match('.*(?= \()', dataset_folder)
        if not m:
            continue
        
        dataset_name = m.group(0)
        folder_of_dataset[dataset_name] = dataset_folder
        
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
        
        for graph_file in list_files(path_to_graphs_of_cur_class):
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
    
    
def load_dataset(datasets_path, dataset):
    folder_of_dataset = determine_folder_of_dataset_dict(datasets_path)
    
    datasets = folder_of_dataset.keys()
    if not dataset in datasets:
        print '%s is not a valid dataset name.' % dataset
        sys.exit(1)
    
    classes_path = join(datasets_path, folder_of_dataset[dataset], 'pz')

    folder_of_class = determine_folder_of_class_dict(classes_path)

    return determine_graph_of_num_dict(classes_path, folder_of_class)