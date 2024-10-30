# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:39:13 2024

@author: isaac
"""

#Import initial modules
import argparse
import numpy as np
import networkx as nx
import pandas as pd
import geopandas as gpd
import matplotlib as plt
import os
import tarfile

#################################
###### DATA PRE-PROCESSING ######
#################################

def read_adjacency_matrix(filename):
    adjacency_matrix = np.load(filename)
    return adjacency_matrix

