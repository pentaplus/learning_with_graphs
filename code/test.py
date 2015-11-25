from conversion import pz
import networkx as nx
import os
import scipy
import scipy.sparse.csr


# import os.path
android_fcg_path = os.path.join("..", "datasets", ("ANDROID FCG (2 classes, "
                                "26 directed graphs, unlabeled edges)"))

fcg_clean_path = os.path.join(android_fcg_path, "clean")
fcg_mal_path = os.path.join(android_fcg_path, "malware")

os.listdir(".")

clean_files = os.listdir()
clean_files
type(clean_files)

first_app_name = clean_files[0]
first_app_name
type(first_app_name)
first_app_name

first_app = pz.load(os.path.join(fcg_clean_path, first_app_name))
type(first_app)

first_app


# information about graph structure
# G = nx.DiGraph(first_app) # new construction not necessary
G = first_app

# number of nodes
G.order()
G.number_of_nodes()
# number of edges
G.size()
G.number_of_edges()
# number of selfloop edges
G.number_of_selfloops()


# adjacency matrix of G
am = nx.adjacency_matrix(G)
type(am)
am.ndim
am.shape
am_dense = am.todense()


# degree dictionary of G (keys = nodes, values = degrees of the corresponding
# nodes)
degreeDict = G.degree()
type(degreeDict)
len(degreeDict)
degreeDict_keys = degreeDict.keys()
degreeDict_keys[0:3]
type(degreeDict_keys[0])
type(degreeDict_keys[0][0])


# iterator over the nodes
nodes_iter = G.nodes_iter()
type(nodes_iter)
for key in nodes_iter:
    print key
    

# list of nodes in the graph
nodes = G.nodes()
print len(nodes)


# first node
first_node = nodes[0]
type(first_node)
print first_node
first_entry_in_first_node = first_node[0]
type(first_entry_in_first_node)
print first_entry_in_first_node

tup = ('Lorg/telegram/SQLite/SQLiteDatabase;', 'getSQLiteHandle', '()I')
tup in nodes
tup in G

G.neighbors(tup)


# find node with at least one neighbor
i = 0
while True:
    if G.neighbors(nodes[i]) != []:
        break
    else:
        i += 1

print i
print nodes[i]
print G.neighbors(nodes[i])
print len(G.neighbors(nodes[i]))

# find node with at least two neighbors
i = 0
while True:
    if len(G.neighbors(nodes[i])) >= 2:
        break
    else:
        i += 1
        
print i

tup7 = nodes[7]
print tup7
G.neighbors(tup7)
len(G.neighbors(tup7))
tup7 in G


# list of edges
edges = G.edges()
first_edge = edges[0]
first_edge
edge = (('Landroid/support/v4/view/KeyEventCompat$EclairKeyEventVersionImpl;',
  '<init>',
  '()V'),
 ('Landroid/support/v4/view/KeyEventCompat$BaseKeyEventVersionImpl;',
  '<init>',
  '()V'))
  
edge in edges # True
edge in G # False, since nodes are considered in this query

tup7
G.degree(tup7)
G.neighbors(tup7)
G.edges([tup7])
len(G.edges([tup7]))
G.in_edges([tup7]) # []
G.out_edges([tup7])

# verify that the graph is directed
second_node = nodes[1]
second_node

neighbors_of_second_node = G.neighbors(second_node)
first_neighbor_of_second_node = neighbors_of_second_node[0]
first_neighbor_of_second_node

second_node
G.neighbors(first_neighbor_of_second_node)

(second_node, first_neighbor_of_second_node) in G.edges() # True
(first_neighbor_of_second_node, second_node) in G.edges() # False


# fetching all clean graphs
# All of them are directed graphs
graphs_ben = []
files_ben = os.listdir(fcg_clean_path)
for g_file in files_ben:
    g_ben = pz.load(os.path.join(fcg_clean_path, g_file))
    g_ben = nx.DiGraph(g_ben) # unnecessary, just for better use with IDE
    graphs_ben.append(g_ben)

node_counts_ben = {}   
for g_ben in graphs_ben:
    node_counts_ben[g_ben] = g_ben.number_of_nodes()

print node_counts_ben
g_ben_max = max(node_counts_ben, key = node_counts_ben.get)
g_ben_max.number_of_nodes()
    
for g_ben in graphs_ben:
    print type(g_ben)
    
first_graph_ben = graphs_ben[0]
first_graph_ben.number_of_nodes()


# fetching all malware graphs
# All of them are directed graphs
graphs_mal = []
files_mal = os.listdir(fcg_mal_path)
for g_file in files_mal:
    g_mal = pz.load(os.path.join(fcg_mal_path, g_file))
    g_mal = nx.DiGraph(g_mal) # unnecessary, just for better use with IDE
    graphs_mal.append(g_mal)
    
first_graph_mal = graphs_mal[0]
first_graph_mal.number_of_nodes()
    
node_counts_mal = {}   
for g_mal in graphs_mal:
    node_counts_mal[g_mal] = g_mal.number_of_nodes()

print node_counts_mal
g_mal_max = max(node_counts_mal, key = node_counts_mal.get)
g_mal_max.number_of_nodes()
    
for g_mal in graphs_mal:
    print type(g_mal) 

print g_ben_max.number_of_edges()
print g_mal_max.number_of_edges()


