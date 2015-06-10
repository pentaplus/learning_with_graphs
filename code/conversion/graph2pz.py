import networkx as nx
import pz
import os


def read_line(fid):
    return fid.readline().rstrip()    

def read_node_labels(node_labels_line):
    # splitting on whitespaces
    return node_labels_line.split()



files = os.listdir('.')

graph_numbers = []
for f in files:
    first_index_of_file_ext = f.find('.graph')
    if first_index_of_file_ext == -1:
        continue
    
    file_name = f[0:first_index_of_file_ext]
    if file_name.isdigit():
        graph_numbers.append(int(file_name))
        

for n in graph_numbers:
    print('Converting graph no ' + str(n))

    fid = open(str(n) + '.graph', 'r')
    
    # -------------------------------------------------------------------------
    # parse graph file
    # -------------------------------------------------------------------------    
    while True:
        cur_line = read_line(fid)
        if cur_line == 'node labels':
            break
        
    # cur_line == 'node labels'
        
    node_labels_line = read_line(fid)
    node_labels = read_node_labels(node_labels_line)
    
    
    while True:
        cur_line = read_line(fid)
        if cur_line == 'adjacency list':
            break
    
    # cur_line == 'adjacency list'
    
    adjacency_list = []
    while True:
        # for MUTAG, NCI1 and NCI109
        cur_line = read_line(fid)
        if cur_line == 'edge labels':
#        # for DD and ENZYMES
#        cur_line = fid.readline()
#        if cur_line == '':
            break
        else:
            cur_line = cur_line.rstrip()            
            cur_neighbor_list = cur_line.split()
            cur_neighbors = [int(i) for i in cur_neighbor_list]
            adjacency_list.append(cur_neighbors)

#    # for DD and ENZYMES
#    # cur_line == '', that is eof is reached
#    fid.close()
            
    cur_line == 'edge labels'
    
#    # for MUTAG
#    edge_list = []
#    while True:
#        cur_line = fid.readline()
#        if cur_line != '':
#            cur_line = cur_line.rstrip() 
#            cur_edge = cur_line.split()        
#            if len(cur_edge) == 3:
#                edge_list.append((int(cur_edge[0]), int(cur_edge[1]),
#                cur_edge[2]))
#        else:
#            break    
    
    # for NIC1 and NCI109            
    weight_list = []
    while True:
        cur_line = fid.readline()
        if cur_line != '':
            cur_line = cur_line.rstrip()
            cur_weight_list = cur_line.split()
            cur_weights = [int(i) for i in cur_weight_list]
            weight_list.append(cur_weights)
        else:
            break
    
    # cur_line == '', that is eof is reached        
    fid.close()
    
    # -------------------------------------------------------------------------
    # create a networkx graph corresponding to the parsed graph
    # -------------------------------------------------------------------------
    
    # create an empty graph (according to the description of the
    # datasets the graphs are undirected)        
    G = nx.Graph()
    
    # add nodes to the graph
    nodes_count = len(node_labels)
    
    for i in xrange(nodes_count):
        G.add_node(i, label = node_labels[i])
    
    
    # add edges to the graph
    
#    # for DD and ENZYMES
#    nodes_count = len(adjacency_list)
#    for i in xrange(nodes_count):
#        cur_neighbors = adjacency_list[i]
#        for j in cur_neighbors:
#            G.add_edge(i, j - 1)
            
    # for NIC1 and NCI109
    if len(adjacency_list) != len(weight_list):
        print('len(adjacency_list) != len(weight_list)')
        exit()
        
    nodes_count = len(adjacency_list)
    for i in xrange(nodes_count):
        cur_neighbors = adjacency_list[i]
        try:
            cur_weights = weight_list[i]
        except IndexError:
            print('hmm')
            
        if len(cur_neighbors) != len(cur_weights):
            print('len(cur_neighbors) != len(cur_weights)')
            exit()
        
        cur_neighbors_count = len(cur_neighbors)
        for j in xrange(cur_neighbors_count):
            G.add_edge(i, cur_neighbors[j] - 1, weight = cur_weights[j]) 
    
    
    
#    # for MUTAG    
#    edges_count = len(edge_list)
#    for i in xrange(edges_count):
#        G.add_edge(edge_list[i][0] - 1, edge_list[i][1] - 1,
#        weight = edge_list[i][2])
    
        
    pz.save(G, str(n) + ".pz")






# test section -----------------------------------------
import pz
G = pz.load("2680.pz")
G.nodes(data = True)
 
G.number_of_edges()   

for i in xrange(G.number_of_nodes()):
    print(G.edges(i, data = True))



