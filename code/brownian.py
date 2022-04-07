#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 17:01:03 2022

@author: fabien
"""

import numpy as np
from scipy.spatial import distance as dist


c1 = np.random.random((5,2))
c2 = np.random.random((7,2))

DIST = dist.cdist(c1, c2, 'euclidean')