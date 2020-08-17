#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 15:43:05 2020
@author: fabien
https://media4.obspm.fr/public/M2R/appliquettes/Turing/AutomateTuring_website.html
"""
import numpy as np, pylab as plt
import cv2
from skimage import filters as ft
from matplotlib import colors

############################################### FONCTION & CLASS
def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv//3], 16) for i in range(0, lv, lv//3))

def img_voronoi_refresh(img, facets, cells_):
    for i in range(0,len(facets)) :
        ifacet = facets[i].astype(np.int)
        if cells_[i] == 1 : color = (1,0,0)
        else : color = (0,0,0)
        #change values of img_voronoi array
        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0)    

class plotting_ion():
    def __init__(self, cells_img, c1, c2):
        self.cmap_ = colors.ListedColormap(['#'+c1, '#'+c2])
        plt.ion()
        size = np.array(cells_img.shape)
        self.dpi, alpha = 360.0, 3
        figsize= alpha*(size[1]/float(self.dpi)), alpha*(size[0]/float(self.dpi))
        self.fig = plt.figure(figsize=figsize, dpi=self.dpi, facecolor="white")
        self.fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
        self.im = plt.imshow(cells_img, interpolation='gaussian', cmap=self.cmap_)
        plt.xticks([]), plt.yticks([])
    
    def plot_and_savefig(self, cells_img, t):
        self.im.set_data(cells_img)
        self.im.set_clim(vmin=cells_img.min(), vmax=cells_img.max())
        plt.draw()
        # To make movie
        plt.savefig("DATA/%03d.png" % t ,dpi=self.dpi)

############################################### PARAMETER
# Grid size & Nb iteration time
N, Time = 128, 25  # N = 64 -> 256 (test -> final)
# Action Radius
Ra, Ri = 1.05, 5. # activator, inhibitor
# Flux diffusion
Ja, Ji = 1., -0.1 # activator, inhibitor
# Offset (palier d'activation)
h = 0
# Bravais (orthorombic or hexagonal)
L_shift = np.sqrt(1-(1/2)**2) #hexagonal, 0.5 for orthorombic
#color polygon
color1, color2 = "4c84bf", "254469"
c0, c1 = hex_to_rgb(color1), hex_to_rgb(color2)

############################################### GRID Construction
# Centered Hexagonal Grid Coordinate (=! orthorombic centered)
hex_grid = np.mgrid[0:N:L_shift, 0:N:1] # [X, Y]
hex_grid[1,::2,:] = hex_grid[1,::2,:]+0.5 # Y translation ::2
# Linearisation
xy = hex_grid.reshape(2,-1).T

############################################### RADIUS
# Coordinate inclusion (euclidean dist)
Coor_r1, Coor_r2 = [],[]
for i in range(len(xy)) :
    d = np.linalg.norm(xy - xy[i], axis=1)
    Coor_r1 += [np.where(d<Ra)[0]]
    Coor_r2 += [np.where((d>=Ra)*(d<=Ri))[0]]
# Array conversion and storage
coor_r1, coor_r2 = np.array(Coor_r1), np.array(Coor_r2)

############################################### RESOLUTION
# C.I.
cells = np.random.randint(-1,1,len(xy))
cells[cells == 0] = 1
CELLS = [cells]
# Algo
for t in np.arange(Time):
    new_cells = np.zeros(len(xy))
    for c in np.arange(len(xy)) :
        new_cells[c] = np.sign(h + Ja*np.sum(cells[coor_r1[c]]) + Ji*np.sum(cells[coor_r2[c]]))
    new_cells[new_cells==0] = 1
    cells = new_cells
    CELLS += [cells]

############################################### VORONOI && IMSAVE
"""
Speed : cv2 > scipy.spatial && cv2 > plt.fill
"""
# Space definition
subdiv  = cv2.Subdiv2D((0, 0, N, N))
# coordinate insertion
for p in xy :
    subdiv.insert((p[0], p[1]))
# get alls face
(facets, centers) = subdiv.getVoronoiFacetList([])
# img voronoi construct
img_voronoi = np.zeros((N,N))

# Draw voronoi diagram
img_voronoi_refresh(img_voronoi, facets, CELLS[0])

# plot initialisation
plot = plotting_ion(img_voronoi, color1, color2)
plot.plot_and_savefig(ft.gaussian(img_voronoi,1), 0)

# ion plot loop
t = 1
for c in CELLS[1:]:
    img_voronoi_refresh(img_voronoi, facets, c)
    plot.plot_and_savefig(ft.gaussian(img_voronoi,1), t)
    t+=1
