# 1. Schauen, wie die logarithmische Zeit-Skala aussieht
# 2. Höhen eintragen

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
import numpy as np
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
FLASH_CFG = 'FLASH CFG'
ANDROID_FCG = 'ANDROID FCG'

EMBEDDING_ABBRVS = {
    WEISFEILER_LEHMAN: 'WL',
    NEIGHBORHOOD_HASH: 'NH',
    COUNT_SENSITIVE_NEIGHBORHOOD_HASH: 'CSNH',
    COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER: 'CSNH ALL',
    GRAPHLET_KERNEL_3: '3-GK',
    GRAPHLET_KERNEL_4: '4-GK',
    RANDOM_WALK_KERNEL: 'RW',
    EIGEN_KERNEL: 'EGK'
    }
    
FONT_SIZE = 10
LEGEND_FONT_SIZE = 8
    
SMALL = 'small'
LARGE = 'large'
    
DATASET_TYPES = [SMALL, LARGE]


mpl.use("pgf")
pgf_with_rc_fonts = {
    "font.family": "Minion Pro",
#    "font.serif": [],                   # use latex default serif font
#    "font.sans-serif": ["DejaVu Sans"], # use a specific sans-serif font
    "pgf.texsystem": "xelatex",
    "text.fontsize": FONT_SIZE
}
mpl.rcParams.update(pgf_with_rc_fonts)


# must be imported after the specification of the RC parameters
import matplotlib.pyplot as plt


# The data matrices DATA_SD and DATA_LD have the following columns:
# embedding name, dataset, score, standard deviation, runtime
DATA_SD = np.array(
    [[WEISFEILER_LEHMAN, MUTAG, 80.0],
     [WEISFEILER_LEHMAN, PTC_MR, 80.0],
     [WEISFEILER_LEHMAN, ENZYMES, 80.0],
     [NEIGHBORHOOD_HASH, MUTAG, 80.0],
     [NEIGHBORHOOD_HASH, PTC_MR, 80.0],
     [NEIGHBORHOOD_HASH, ENZYMES, 80.0],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH, MUTAG, 80.0],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH, PTC_MR, 80.0],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH, ENZYMES, 80.0],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER, MUTAG, 80.0],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER, PTC_MR, 80.0],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER, ENZYMES, 80.0],
     [GRAPHLET_KERNEL_3, MUTAG, 80.0],
     [GRAPHLET_KERNEL_3, PTC_MR, 80.0],
     [GRAPHLET_KERNEL_3, ENZYMES, 80.0],
     [GRAPHLET_KERNEL_4, MUTAG, 80.0],
     [GRAPHLET_KERNEL_4, PTC_MR, 80.0],
     [GRAPHLET_KERNEL_4, ENZYMES, 80.0],
     [RANDOM_WALK_KERNEL, MUTAG, 80.0],
     [RANDOM_WALK_KERNEL, PTC_MR, 80.0],
     [RANDOM_WALK_KERNEL, ENZYMES, 80.0],
     [EIGEN_KERNEL, MUTAG, 80.0],
     [EIGEN_KERNEL, PTC_MR, 80.0],
     [EIGEN_KERNEL, ENZYMES, 80.0]])
     

DATA_LD = np.array(
    [[WEISFEILER_LEHMAN, DD, 80.0],
     [WEISFEILER_LEHMAN, NCI1, 80.0],
     [WEISFEILER_LEHMAN, NCI109, 80.0],
     [WEISFEILER_LEHMAN, FLASH_CFG, 80.0],
     [WEISFEILER_LEHMAN, ANDROID_FCG, 80.0],
     [NEIGHBORHOOD_HASH, DD, 80.0],
     [NEIGHBORHOOD_HASH, NCI1, 80.0],
     [NEIGHBORHOOD_HASH, NCI109, 80.0],
     [NEIGHBORHOOD_HASH, FLASH_CFG, 80.0],
     [NEIGHBORHOOD_HASH, ANDROID_FCG, 80.0],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH, DD, 80.0],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH, NCI1, 80.0],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH, NCI109, 80.0],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH, FLASH_CFG, 80.0],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH, ANDROID_FCG, 80.0],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER, DD, 80.0],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER, NCI1, 80.0],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER, NCI109, 80.0],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER, FLASH_CFG, 80.0],
     [COUNT_SENSITIVE_NEIGHBORHOOD_HASH_ALL_ITER, ANDROID_FCG, 80.0],
     [GRAPHLET_KERNEL_3, DD, 80.0],
     [GRAPHLET_KERNEL_3, NCI1, 80.0],
     [GRAPHLET_KERNEL_3, NCI109, 80.0],
     [GRAPHLET_KERNEL_3, FLASH_CFG, 80.0],
     [GRAPHLET_KERNEL_3, ANDROID_FCG, 80.0],
     [GRAPHLET_KERNEL_4, DD, 80.0],
     [GRAPHLET_KERNEL_4, NCI1, 80.0],
     [GRAPHLET_KERNEL_4, NCI109, 80.0],
     [GRAPHLET_KERNEL_4, FLASH_CFG, 80.0],
     [GRAPHLET_KERNEL_4, ANDROID_FCG, 80.0],
     [RANDOM_WALK_KERNEL, DD, 80.0],
     [RANDOM_WALK_KERNEL, NCI1, 80.0],
     [RANDOM_WALK_KERNEL, NCI109, 80.0],
     [RANDOM_WALK_KERNEL, FLASH_CFG, 80.0],
     [RANDOM_WALK_KERNEL, ANDROID_FCG, 80.0],
     [EIGEN_KERNEL, DD, 80.0],
     [EIGEN_KERNEL, NCI1, 80.0],
     [EIGEN_KERNEL, NCI109, 80.0],
     [EIGEN_KERNEL, FLASH_CFG, 80.0],
     [EIGEN_KERNEL, ANDROID_FCG, 80.0]])
     
      
# order according to the sequence of the embeddings in the data matrices
COLORS = ['#00008F', '#0020FF', '#00AFFF', '#40FFBF', '#CFFF30', '#FF9F00',
          '#FF1000', '#800000']
         

for dataset_type in DATASET_TYPES:         
         
    DATA = DATA_SD if dataset_type == SMALL else DATA_LD
    yerr = range(3) if dataset_type == SMALL else range(5)
    
    
    fig = plt.figure(figsize = (5.8, 3))
    ax = fig.add_subplot(111)
    
    
    # embeddings ordered according to their sequence in the data matrices
    indices = np.unique(DATA[:,0], return_index = True)[1]
    embeddings = [DATA[:,0][index] for index in sorted(indices)]
    
    # datasets ordered according to their sequence in the data matrices
    indices = np.unique(DATA[:,1], return_index = True)[1]
    datasets = [DATA[:,1][index] for index in sorted(indices)]
    #scores = DATA[:,2]
    #std_devs = DATA[:,3]
    #runtimes = DATA[:,4]
    
    #num_embeddings = len(embeddings)
    
    space = 2/(len(embeddings) + 2)
    
    width = (1 - space) / len(embeddings)
    print "width:", width
    
    for i, embedding in enumerate(embeddings):
        print "embedding:", embedding
        vals = DATA[DATA[:,0] == embedding][:,2].astype(np.float)
        pos = [j - (1 - space)/2 + i * width for j in range(1, len(datasets) + 1)]
    
        # add label param
        ax.bar(pos, vals, width = width, color = [COLORS[i]] * 3, yerr = yerr,
               ecolor = 'black', label = EMBEDDING_ABBRVS[embeddings[i]])
                      
        x = 0
               
    
    ax.set_xlim(0.5 - space/2, len(datasets) + 0.5 + space/2)
    ax.set_ylim([0, 110])
    
    
    # Drawing the canvas causes the labels to be positioned, which is necessary
    # in order to get their values
    fig.canvas.draw()
    xtick_labels = [item.get_text() for item in ax.get_xticklabels()]
    
    x = range(1, len(datasets) + 1)
    #plt.xticks(x, datasets, fontsize = FONT_SIZE)
    plt.xticks(x, datasets)
    
    #if dataset_type == SMALL:
    #    xtick_labels[1] = u''
    #    xtick_labels[2] = u'MUTAG'
    #    xtick_labels[3] = u''
    #    xtick_labels[4] = u'PTC(MR)'
    #    xtick_labels[5] = u''
    #    xtick_labels[6] = u'ENZYMES'
    #    xtick_labels[7] = u''
    #else:
    #    # dataset_type == LARGE
    #    pass
    
    [u'', u'1', u'2', u'3', u'4', u'5', u'']
    
    #ax.set_xticklabels(xtick_labels)
    
    
    # plot legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, prop = {'size': LEGEND_FONT_SIZE})
    #ax.legend(handles, labels)
    #plot.legend(bbox_to_anchor = (1.2, 0.5))
    
    plt.tight_layout(0.5)
    
    output_file_name = 'figure_' + dataset_type
    plt.savefig(output_file_name + '.pdf')
    plt.savefig(join(TARGET_PATH, output_file_name + '.pgf'))


