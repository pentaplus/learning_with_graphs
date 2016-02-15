import inspect
import sys
import time

from numpy import mean
from os.path import abspath, dirname, join

# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))

from misc import dataset_loader



t0 = time.time()


DATASETS_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', 'datasets')

#DATASET = 'ANDROID FCG' # !! change file names from hashes to numbers
#DATASET = 'CFG' # !! change file names from hashes to numbers
#DATASET = 'DD'
#DATASET = 'ENZYMES'
#DATASET = 'MUTAG'
#DATASET = 'NCI1'
#DATASET = 'NCI109'
DATASET = 'PTC(MR)'


graph_of_num, class_lbls = dataset_loader.load_dataset(DATASETS_PATH, DATASET)
    
graphs_of_class = dataset_loader.determine_graphs_of_class_dict(graph_of_num)

classes = graphs_of_class.keys()

# calculate statistics
node_counts = []
edge_counts = []
degrees = []
min_deg = float("inf")
max_deg = 0
number_of_isolated_nodes = 0
for graph, class_number in graph_of_num.itervalues():
    node_counts.append(graph.number_of_nodes())
    edge_counts.append(graph.number_of_edges())
    degrees.append(mean(graph.degree().values()))
    
    if min(graph.degree().values()) < min_deg:
        min_deg = min(graph.degree().values())
        
    if max(graph.degree().values()) > max_deg:
        max_deg = max(graph.degree().values())
        
    for degree in graph.degree().values():
        if degree == 0:
           number_of_isolated_nodes += 1 

avg_v = mean(node_counts)
avg_e = mean(edge_counts)
max_v = max(node_counts)
max_e = max(edge_counts)
min_v = min(node_counts)
avg_deg = mean(degrees)

print 'dataset:', DATASET
print '# graphs:', len(graph_of_num)
print '# classes:', len(classes)

for class_number in graphs_of_class.iterkeys():
    print 'class %d: %d' % (class_number, len(graphs_of_class[class_number]))

print 'avg_v: %.2f' % avg_v
print 'avg_e: %.2f' % avg_e
print 'max_v:', max_v
print 'max_e:', max_e
print 'min_v:', min_v
print 'avg_deg: %.3f' % avg_deg
print 'max_deg:', max_deg
print 'min_deg:', min_deg
print 'isolated:', number_of_isolated_nodes, '\n'


t1 = time.time()
total = t1 - t0

print "The execution took %.2f seconds." % total

