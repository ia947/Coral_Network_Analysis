# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:01:43 2024

@author: isaac
"""

import numpy as np
import os
import tarfile

# Extracting .tar file
io_tar_file_path = "IOMAtrices.tar.gz"
with tarfile.open(io_tar_file_path, 'r:gz') as tar:
    tar.extractall()
