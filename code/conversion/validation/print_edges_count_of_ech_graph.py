import inspect
import sys

from os.path import abspath, dirname, join

# determine script path
filename = inspect.getframeinfo(inspect.currentframe()).filename
script_path = dirname(abspath(filename))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(script_path, '..', '..'))

from misc import datasetloader


DATASETS_PATH = join(script_path, '..', '..', '..', 'datasets')
# dataset = 'MUTAG'
# dataset = 'DD'
dataset = 'ENZYMES'
# dataset = 'NCI1'
# dataset = 'NCI109'

graph_of_num = datasetloader.load_dataset(DATASETS_PATH, dataset)

f = open('python_edges_count_of_each_graph.csv', 'w')    
for graph_num, (G, class_lbl) in graph_of_num.iteritems():
	f.write(str(graph_num) + '; ' + str(2*G.number_of_edges()) + '\n')
f.close()