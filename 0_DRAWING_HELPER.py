#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:33:06 2021
@author: fabien
"""
import cv2,os, scipy as sp
import numpy as np, pylab as plt

from sklearn import preprocessing
from sklearn.cluster import Birch

from matplotlib.collections import PolyCollection
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import cascaded_union

################################### PARAMETER
FILENAME = "1_Portrait.jpg"
NB_COLOR = 32
BLUR = 2
THRESH = 2
DILATION = 1
SIMPLE = 1

################################### CLUSTERING PART
# img import
img_original = cv2.imread(os.getcwd() + os.path.sep + FILENAME)
H,L,C = img_original.shape
# Normalize
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(img_original.reshape(-1,3))    
# Clustering (find max color number)
cluster = Birch(n_clusters = NB_COLOR, threshold=X.std()/np.pi).fit(X)
# level image
img_cluster = cluster.labels_.reshape(H,L)
# blur
img_cluster_blur = cv2.blur(img_cluster,(BLUR,BLUR)).astype(np.uint8)
# Manual contouring (Canny method)
img_border = cv2.Canny(img_cluster_blur, THRESH,THRESH)
# Coordinate
x,y = np.where(img_border != 0)
COOR = np.concatenate((x[:,None], y[:,None]), axis = 1)
# Triangulation
Tri = sp.spatial.Delaunay(COOR)
coor_tri = COOR[Tri.simplices]
# center of each triangle
tri_center = np.rint(np.mean(coor_tri, axis=1)).astype(int)
tri_center = tuple(map(tuple, tri_center.T))
# cluster color for each triangle
tri_color = img_cluster[tri_center]
print("Clustering Triangulation Done")
################################### MERGING POLY PART
# convert triangle to shapely format
multipolygon, poly_color = [], []
for i in range(tri_color.max()+1) :
    LOC_TRI = np.where(tri_color == i)
    # to polygon
    polygons = map(Polygon, coor_tri[LOC_TRI])
    # to multi with dilate polygon
    multipolygon += [MultiPolygon(polygons).buffer(DILATION)]
    poly_color += [i]
    print("Extract Polygon and Dilation : " +str(i) + "/" + str(tri_color.max()))
# union multi-polygon
merge_poly, merge_color = [], []
for m,c in zip(multipolygon,poly_color) :
    union_poly = cascaded_union(m)
    # verify type
    if isinstance(union_poly, Polygon) : 
        union_poly = MultiPolygon([union_poly])
    # listing with simplify
    merge_poly += [union_poly.simplify(SIMPLE)]
    merge_color += [np.asarray([c]*len(union_poly))]
    print("Union Polygon and Simplification : " + str(c) + "/" + str(max(poly_color)))
merge_color = np.concatenate(merge_color)
# Extract ext_coor in numpy array
new_coor, area_p = [], []
for mp in merge_poly :
    for p in mp :
        area_p += [p.area]
        new_coor += [np.array(p.exterior.coords.xy).T]
################################### COLORING  PART
# color construction
POLY_color_RGB = np.zeros((merge_color.shape[0],3)).astype(np.float32)
for i in range(merge_color.max()+1):
    LOC_IMG = np.where(img_cluster == i)
    COLOR = img_original[LOC_IMG].mean(axis=0)
    LOC_TRI = np.where(merge_color == i)
    POLY_color_RGB[LOC_TRI] = COLOR
POLY_color_RGBA = cv2.cvtColor(POLY_color_RGB[:,None,:], cv2.COLOR_BGR2RGBA).squeeze()/255
POLY_color_RGBA[:,-1] = 1
# convert to numpy
POLY_COOR, AREA_POLY = np.array(new_coor), np.array(area_p)
# ordering
order_idx = np.argsort(AREA_POLY)[::-1]
AREA_POLY = AREA_POLY[order_idx]
POLY_COOR = POLY_COOR[order_idx]
POLY_color_RGBA = POLY_color_RGBA[order_idx]      
# filtering
POLY_COOR = POLY_COOR[AREA_POLY > 50]
POLY_color_RGBA = POLY_color_RGBA[AREA_POLY > 50] 
################################### TO SVG PART
## Figures
fig = plt.figure(figsize=(H/100,L/100), dpi=120) 
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
plt.xticks([]), plt.yticks([])
plt.xlim(0,H); plt.ylim(0,L)
poly = PolyCollection(POLY_COOR, lw = 0., edgecolors=POLY_color_RGBA, facecolors=POLY_color_RGBA)
ax.add_collection(poly)
plt.savefig('2_DRAWING_AUTO.svg')
plt.show()    
