import networkx as nx
import numpy as np
import os
import pz
import shutil
import sys

# D&D dataset
#number_of_graphs = 1178
#
#path_class_1 = os.path.join('DD_pz', 'class 1')
#path_class_2 = os.path.join('DD_pz', 'class 2')
#
#graphs = []
#for graph_no in xrange(1, number_of_graphs + 1):
#    file_name = str(graph_no) + '.pz'
#    
#    if os.path.isfile(os.path.join(path_class_1, file_name)):
#        print('graph ' + str(graph_no) + ' belongs to class 1.')
#        G = pz.load(os.path.join(path_class_1, file_name))
#    elif os.path.isfile(os.path.join(path_class_2, file_name)):
#        print('graph ' + str(graph_no) + ' belongs to class 2.')
#        G = pz.load(os.path.join(path_class_2, file_name))
#    else:
#        print('graph ' + str(graph_no) + ' does not exist.')
#        sys.exit(1)
#        
#    graphs.append(G)


# ENZYMES dataset
#number_of_graphs = 600
#
#path_class_1 = os.path.join('ENZ_pz', 'class 1')
#path_class_2 = os.path.join('ENZ_pz', 'class 2')
#path_class_3 = os.path.join('ENZ_pz', 'class 3')
#path_class_4 = os.path.join('ENZ_pz', 'class 4')
#path_class_5 = os.path.join('ENZ_pz', 'class 5')
#path_class_6 = os.path.join('ENZ_pz', 'class 6')
#
#graphs = []
#for graph_no in xrange(1, number_of_graphs + 1):
#    file_name = str(graph_no) + '.pz'
#    
#    if os.path.isfile(os.path.join(path_class_1, file_name)):
#        print('graph ' + str(graph_no) + ' belongs to class 1.')
#        G = pz.load(os.path.join(path_class_1, file_name))
#    elif os.path.isfile(os.path.join(path_class_2, file_name)):
#        print('graph ' + str(graph_no) + ' belongs to class 2.')
#        G = pz.load(os.path.join(path_class_2, file_name))
#    elif os.path.isfile(os.path.join(path_class_3, file_name)):
#        print('graph ' + str(graph_no) + ' belongs to class 3.')
#        G = pz.load(os.path.join(path_class_3, file_name))
#    elif os.path.isfile(os.path.join(path_class_4, file_name)):
#        print('graph ' + str(graph_no) + ' belongs to class 4.')
#        G = pz.load(os.path.join(path_class_4, file_name))
#    elif os.path.isfile(os.path.join(path_class_5, file_name)):
#        print('graph ' + str(graph_no) + ' belongs to class 5.')
#        G = pz.load(os.path.join(path_class_5, file_name))
#    elif os.path.isfile(os.path.join(path_class_6, file_name)):
#        print('graph ' + str(graph_no) + ' belongs to class 6.')
#        G = pz.load(os.path.join(path_class_6, file_name))
#    else:
#        print('graph ' + str(graph_no) + ' does not exist.')
#        sys.exit(1)
#        
#    graphs.append(G)

# PTC(MR) dataset
number_of_graphs = 344

path_class_1 = os.path.join('PTC_pz', 'class 1')
path_class_minus_1 = os.path.join('PTC_pz', 'class -1')

graphs = []
for graph_no in xrange(0, number_of_graphs):
    file_name = str(graph_no) + '.pz'
    
    if os.path.isfile(os.path.join(path_class_1, file_name)):
        print('graph ' + str(graph_no) + ' belongs to class 1.')
        G = pz.load(os.path.join(path_class_1, file_name))
    elif os.path.isfile(os.path.join(path_class_minus_1, file_name)):
        print('graph ' + str(graph_no) + ' belongs to class 2.')
        G = pz.load(os.path.join(path_class_minus_1, file_name))
    else:
        print('graph ' + str(graph_no) + ' does not exist.')
        sys.exit(1)
        
    graphs.append(G)
    
    
# G1 = graphs[0]

node_sizes = [G.number_of_nodes() for G in graphs]
edge_sizes = [G.number_of_edges() for G in graphs]
degrees = [np.mean(G.degree().values()) for G in graphs]

avg_v = np.mean(node_sizes)
avg_e = np.mean(edge_sizes)
max_v = max(node_sizes)
max_e = max(edge_sizes)
min_v = min(node_sizes)
avg_deg = np.mean(degrees)

print 'avg_v: ', avg_v
print 'avg_e: ', avg_e
print 'max_v: ', max_v
print 'max_e: ', max_e
print 'min_v: ', min_v
print 'avg_deg: ', avg_deg
