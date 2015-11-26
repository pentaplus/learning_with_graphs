import networkx as nx
import os
import pz

# test section -------------------------------------------------------------------


# --------------------------------------------------------------------------------

android_fcg_path = os.path.join("..", "..", "datasets", ("ANDROID FCG (2 "
                                "classes, 26 directed graphs, "
                                "unlabeled edges)"))
                                
fcg_clean_path = os.path.join(android_fcg_path, "clean")
fcg_mal_path = os.path.join(android_fcg_path, "malware")

os.listdir(fcg_clean_path)

