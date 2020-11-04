#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:10:18 2020
@author: fabien
https://networkx.github.io/documentation/stable/tutorial.html
https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html
"""

import numpy as np, pylab as plt
import networkx as nx, pandas as pd
import matplotlib.animation as animation

############################################### PARAMETER
L = 2.
branch, height= 4, 5
angle_, shift_lr = (50,130), (.5,1./3) #shift lr : golden ratio

init = np.zeros(branch)
theta = np.radians(np.linspace(angle_[0], angle_[1], branch))
shift = np.abs(np.linspace(-shift_lr[0], shift_lr[1], branch))
shift = np.concatenate((shift[:,None], shift[:,None]), axis =1)

############################################### TREE GRAPH CONSTRUCT
btree = nx.balanced_tree(branch, height)
df_tree = nx.to_pandas_edgelist(btree)

# add init node parameter
C = np.zeros((df_tree.shape[0], 7))
df_tree[['L','Depth', 'xA', 'yA', 'xB','yB', 'theta']] = C

# add ground node
df_tree.loc[-1] = {'source' : -1, 'target' : 0, 'L' : L, 'Depth' : 0,
                   'xA': 0, 'yA': 0, 'xB' : 0,'yB' :  L,'theta' :  0}
df_tree.index = df_tree.index + 1
df_tree = df_tree.sort_index()

# calculate position of each node
src_tree = df_tree.groupby('source')
for i, st in src_tree :
    if i > -1 :
        prms = df_tree[df_tree.target == st.source.iloc[0]]
        ind = src_tree.indices[i]
        df_tree.loc[ind, 'L'] = (prms.L.values/2) * np.ones(st.L.shape)
        df_tree.loc[ind, 'Depth'] = (prms.Depth.values + 1) * np.ones(st.L.shape)
        
        #rotation part
        phi, l = prms.theta.values, prms.L.values
        rotMatrix = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi),  np.cos(phi)]]).squeeze()
        x_, y_ = (l/2)*np.cos(theta), (l/2)*np.sin(theta)
        xy = np.concatenate((x_[:,None], y_[:,None]), axis=1).T
        rot_v = np.dot(rotMatrix, xy)
        df_tree.loc[ind, 'theta'] = np.arctan2(rot_v[0],rot_v[1]) # (x,y)
        
        #translation part
        xy_00 = prms[['xB', 'yB']].values - prms[['xA', 'yA']].values
        df_tree.loc[ind, ['xA','yA']] = prms[['xA','yA']].values + (1-shift)*xy_00
        
        #linear combinaison
        df_tree.loc[ind, ['xB', 'yB']] = prms[['xA','yA']].values + (1-shift)*xy_00 + rot_v.T

############################################### PLOT ANIMATION
nodeX, nodeY = df_tree[['xA','xB']].values, df_tree[['yA','yB']].values
lvl =  df_tree['Depth'].values
thickness = pd.Series(2*np.pi*((np.max(lvl) - lvl)/np.max(lvl)) + 1)

list_t = np.repeat(np.arange(height+1)[::-1], 20)

# figure declaration
fig = plt.figure(figsize=(5,5), dpi=120) 
ax = fig.add_subplot(111) #fig.add_axes([0., 0., 1., 1.], frameon=False)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
plt.xticks([]), plt.yticks([])
plt.xlim([-L, L]); plt.ylim([0, L + 4/5 * L])

ax.spines['bottom'].set_color('None'); ax.spines['top'].set_color('None') 
ax.spines['right'].set_color('None'); ax.spines['left'].set_color('None')

# set figure background
fig.patch.set_alpha(0.) # 0. for nothing, 1 for max
fig.patch.set_facecolor("None"); ax.set_facecolor("None") #None for alpha

indices = np.arange(0,len(nodeX))
# Création des noeud/arete à mettre à jour :
line = [ax.plot([],[], color='k')[0] for i in indices]

def animate(i):
    list_lvl = lvl <= list_t[i]
    for j in indices :
        if list_lvl[j] == True :
            line[j].set_data(nodeX[j], nodeY[j])
            line[j].set_linewidth(thickness[j])
        else :
            line[j].set_data([],[])
    return line

anim = animation.FuncAnimation(fig, animate, frames=len(list_t))
anim.save(filename='1_animation.mp4', writer='ffmpeg', fps=30, codec="png") #png for alpha
