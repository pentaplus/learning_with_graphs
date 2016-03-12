# 1. Schauen, wie die logarithmische Zeit-Skala aussieht
# 2. Plots für die Laufzeiten auf den kleinen Datensätzen und Plots für die
#    großen Datensätze erstellen
# 3. Alle 4 Säulen-Diagramme in das pdf-Dokument übernehmen

# 1. Die Höhen der Säulen müssen korrekt angegeben werden.
# 2. Die "error bars" müssen richtig spezifiziert werden.
# 3. Die y-Achse muss richtig beschriftet werden.
# 4. Ein weiterer Plot muss mit den Laufzeiten erstellt werden.
# 5. Alle Schritte müssen wiederholt werden für die anderen 5 Datensätze.
"""
Plot bar charts for the classification accuracy of the embedding methods.
"""
from __future__ import division

__author__ = "Benjamin Plock <benjamin.plock@stud.uni-goettingen.de>"
__date__ = "2016-03-10"


import inspect
import matplotlib as mpl
import sys

from os.path import abspath, dirname, join

# determine script path
SCRIPT_PATH = inspect.getframeinfo(inspect.currentframe()).filename
SCRIPT_FOLDER_PATH = dirname(abspath(SCRIPT_PATH))
# modify the search path for modules in order to access modules in subfolders
# of the script's parent directory
sys.path.append(join(SCRIPT_FOLDER_PATH, '..'))


#=================================================================================
# constants
#=================================================================================
TARGET_PATH = join(SCRIPT_FOLDER_PATH, '..', '..', '..', 'tex', 'figures')

# embeddings
WEISFEILER_LEHMAN = 'weisfeiler_lehman'
NEIGHBORHOOD_HASH = 'neighborhood_hash'
COUNT_SENSITIVE_NEIGHBORHOOD_HASH = 'count_sensitive_neighborhood_hash'
COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER \
    = 'count_sensitive_neighborhood_hash_all_iter'
GRAPHLET_KERNEL_3 = 'graphlet_kernel_3'
GRAPHLET_KERNEL_4 = 'graphlet_kernel_4'
LABEL_COUNTER = 'label_counter'
RANDOM_WALK_KERNEL = 'random_walk_kernel'
EIGEN_KERNEL = 'eigen_kernel'

# datasets
MUTAG = 'MUTAG'
PTC_MR = 'PTC(MR)'
ENZYMES = 'ENZYMES'
DD = 'DD'
NCI1 = 'NCI1'
NCI109 = 'NCI109'
ANDROID_FCG_PARTIAL = 'ANDROID FCG PARTIAL'
FLASH_CFG = 'FLASH CFG'

EMBEDDING_SHORTCUTS = {}


mpl.use("pgf")
pgf_with_rc_fonts = {
    "font.family": "Minion Pro",
#    "font.serif": [],                   # use latex default serif font
#    "font.sans-serif": ["DejaVu Sans"], # use a specific sans-serif font
    "pgf.texsystem": "xelatex",
    "text.fontsize": 12
}
mpl.rcParams.update(pgf_with_rc_fonts)


import matplotlib.pyplot as plt


#plt.figure(figsize = (2.75, 2.5))
#plt.plot(range(5))
##plt.text(2.5, 1, "Avg. deg.")
#plt.xlabel(u"$\\mu$")
#plt.tight_layout(0.5)


#import numpy as np
#
#a=np.array([[3,6,8,9,6],[2,3,4,5,6],[4,5,6,7,8],[3,6,5,8,6],[5,8,8,6,5]])
#df=DataFrame(a, columns=['a','b','c','d','e'], index=[2,4,6,8,10])

#df.plot(kind = 'bar', legend = False, width = 4).legend(
#    bbox_to_anchor = (1.2, 0.5))
#plot = df.plot(kind = 'bar', legend = False, width = 50)
#plot.legend(bbox_to_anchor = (1.2, 0.5))
#
#fig = plot.get_figure()


import numpy as np

# The data matrix has the following columns:
# embedding name, dataset, score, standard deviation, runtime
data = np.array(
    [[WEISFEILER_LEHMAN, MUTAG, 9.97],
     [WEISFEILER_LEHMAN, PTC_MR, 27.31],
     [WEISFEILER_LEHMAN, ENZYMES, 5.77],
     [NEIGHBORHOOD_HASH, MUTAG, 9.97],
     [NEIGHBORHOOD_HASH, PTC_MR, 27.31],
     [NEIGHBORHOOD_HASH, ENZYMES, 5.77],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH, MUTAG, 9.97],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH, PTC_MR, 27.31],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH, ENZYMES, 5.77],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER, MUTAG, 9.97],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER, PTC_MR, 27.31],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER, ENZYMES, 5.77],
     [GRAPHLET_KERNEL_3, MUTAG, 9.97],
     [GRAPHLET_KERNEL_3, PTC_MR, 27.31],
     [GRAPHLET_KERNEL_3, ENZYMES, 5.77],
     [GRAPHLET_KERNEL_4, MUTAG, 9.97],
     [GRAPHLET_KERNEL_4, PTC_MR, 27.31],
     [GRAPHLET_KERNEL_4, ENZYMES, 5.77],
     [RANDOM_WALK_KERNEL, MUTAG, 9.97],
     [RANDOM_WALK_KERNEL, PTC_MR, 27.31],
     [RANDOM_WALK_KERNEL, ENZYMES, 5.77],
     [EIGEN_KERNEL, MUTAG, 9.97],
     [EIGEN_KERNEL, PTC_MR, 27.31],
     [EIGEN_KERNEL, ENZYMES, 5.77]])
      
# order according to the sequence of the embeddings in data      
COLORS = ['#00008F', '#0020FF', '#00AFFF', '#40FFBF', '#CFFF30', '#FF9F00',
          '#FF1000', '#800000']
          
LABELS = [WEISFEILER_LEHMAN, NEIGHBORHOOD_HASH, COUNT_SENSITIVE_NEIGHBORHOOD_HASH,
          COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER, GRAPHLET_KERNEL_3,
          GRAPHLET_KERNEL_4, RANDOM_WALK_KERNEL, EIGEN_KERNEL]

fig = plt.figure()
ax = fig.add_subplot(111)


embeddings = np.unique(data[:,0])
datasets = np.unique(data[:,1])
#scores = data[:,2]
#std_devs = data[:,3]
#runtimes = data[:,4]

#num_embeddings = len(embeddings)

space = 2/(len(embeddings) + 2)

width = (1 - space) / len(embeddings)
print "width:", width

for i, embedding in enumerate(embeddings):
    print "embedding:", embedding
    vals = data[data[:,0] == embedding][:,2].astype(np.float)
    pos = [j - (1 - space)/2 + i * width for j in range(1, len(datasets) + 1)]

    # add label param
    rect = ax.bar(pos, vals, width = width, color = [COLORS[i]] * 3, yerr = [1, 2, 3],
                  ecolor = 'black', label = LABELS[i])
                  
    x = 0
           

ax.set_xlim(0.5 - space/2, len(datasets) + 0.5 + space/2)
ax.set_ylim([0, 110])


# Drawing the canvas causes the labels to be positioned, which is necessary
# in order to get their values
fig.canvas.draw()
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[1] = u''
labels[2] = u'MUTAG'
labels[3] = u''
labels[4] = u'PTC(MR)'
labels[5] = u''
labels[6] = u'ENZYMES'
labels[7] = u''

ax.set_xticklabels(labels)

#fig.legend()

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, prop = {'size': 6})

#ax.legend([rect])


plt.savefig('figure.pdf')
plt.savefig(join(TARGET_PATH, 'figure.pgf'))
