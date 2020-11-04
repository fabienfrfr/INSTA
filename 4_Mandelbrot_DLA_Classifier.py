#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 18:04:10 2020
@author: fabien
"""

import numpy as np, pylab as plt
from scipy.spatial import distance
import matplotlib.animation as animation
from sklearn.neighbors import KNeighborsClassifier

#################################### PARAMETER
WALKER_NUMBER = 750
DELTA = np.radians(45) # random walk : 0 (discret) -> 90 (continious)
ALPHA = 0.95 # position of new walker
CONTACT_RADIUS = 0.025

#################################### INITIALISATION
# Position of seed
aggregate = np.array([[0.5,1.]])
# Position of end
seed_end = np.array([[.3,.2]])
# First position of all random walker
walker = np.random.random((WALKER_NUMBER,2))
# Network init
graph = np.array([[-1,0]]) # index, parent

#################################### D.L.A. ALGORITHME
walker_list = [walker.copy()]
aggregation = [aggregate.copy()]
# mvt step for all walker
while not(np.any(np.linalg.norm(aggregate - seed_end, axis=1) < CONTACT_RADIUS)) :
    # Random walk angles
    way = 2*np.pi*(np.random.randint(0,4,WALKER_NUMBER)/4.)
    var = DELTA*np.random.random(WALKER_NUMBER) - DELTA/2
    step_angles = way + var
    # Mvt of each random walker
    z = CONTACT_RADIUS*np.exp(1j*step_angles)
    mvt = np.concatenate((z.real[:,None], z.imag[:,None]), axis = 1)
    # New position of all walker
    new_walker = walker + mvt
    # Outlier
    out_idx = np.where(np.any(new_walker < 0, axis =1) + np.any(new_walker > 1, axis=1))[0]
    new_walker = np.delete(new_walker, out_idx, axis=0)
    new_rand = np.random.random((len(out_idx),2))
    new_rand[:,1] = ALPHA*aggregate[:,1].min()*new_rand[:,1]
    new_walker = np.concatenate((new_walker,new_rand), axis=0)
    # Calculate distance beetwen aggregate and walker
    dist = distance.cdist(aggregate, new_walker)
    parent, daugter = np.where(dist < CONTACT_RADIUS)
    agg_loc, idx = np.unique(daugter, return_index=True)
    # Graph construction
    idx_graph, new_node = np.array(graph[parent[idx],1]), np.arange(np.max(graph)+1,np.max(graph)+len(agg_loc)+1)
    if len(idx_graph > 0) :
        graph_part = np.concatenate((idx_graph[:,None], new_node[:,None]), axis=1)
        graph = np.concatenate((graph, graph_part), axis=0)
    # Add in aggregate and delete in new_walker
    new_agg = new_walker[agg_loc].copy()
    new_walker = np.delete(new_walker, agg_loc, axis=0)
    aggregate = np.concatenate((aggregate,new_agg), axis=0)
    # Adding walker for next loop
    new_rand = np.random.random((len(agg_loc),2))
    new_rand[:,1] = ALPHA*aggregate[:,1].min()*new_rand[:,1]
    walker = np.concatenate((new_walker,new_rand), axis=0)
    # Incrementation
    walker_list += [walker.copy()]
    aggregation += [aggregate.copy()]
T = len(aggregation)

### Seed (start to end) graph reconstruction
parent_idx = np.where(np.linalg.norm(aggregate - seed_end, axis=1) < CONTACT_RADIUS)[0][0]
Lightning = aggregate[parent_idx][None]
while parent_idx != 0 :
    parent_idx = graph[parent_idx][0]
    Lightning = np.concatenate((Lightning, aggregate[parent_idx][None]), axis=0)

### Fitting
PASS = int(WALKER_NUMBER/100)-1
X_1, X_2 = aggregation[-1][::PASS], walker_list[-1][::PASS]
target = np.array(len(X_1)*[0]+len(X_2)*[1]+len(Lightning)*[2])
X_train = np.concatenate((X_1,X_2,Lightning))
# randomize index
permut = np.random.permutation(np.arange(len(target)))
# Fit
clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(X_train[permut], target[permut])
# Prediction & contouring
xx, yy = np.meshgrid(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, Z); plt.savefig('2_DLA.svg'); plt.close()

#################################### PLOT ANIMATION
fig = plt.figure(figsize=(5,5), dpi=120) 
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
plt.xticks([]), plt.yticks([])
plt.xlim(0,1); plt.ylim(0,1)
# Delete border
ax.spines['bottom'].set_color('None'); ax.spines['top'].set_color('None') 
ax.spines['right'].set_color('None'); ax.spines['left'].set_color('None')

# set figure background opacity (alpha) to 0
ax.set_facecolor("None") #None
fig.patch.set_alpha(0.)
fig.patch.set_facecolor("None")#None for alpha

# Création des noeud/arete à mettre à jour :
x_, y_ = walker_list[0].T
a_x, a_y = 10*np.ones(aggregation[-1].T.shape)
point = plt.scatter(x_, y_, marker = '.', lw=0, s=30, alpha=0.25)
agg = plt.scatter(a_x, a_y, marker = '.', lw=0, s=30, alpha=0.5)
end = plt.scatter(seed_end[:,0], seed_end[:,1], marker = '.', lw=0, s=30, alpha=0.25)

# Target seed
line_ = ax.plot([],[], color='white')[0]

def animate(i):
    if i < T-1 :
        xy = aggregation[i]
        a_x[:xy.shape[0]], a_y[:xy.shape[0]] = xy.T
        agg.set_offsets(np.concatenate((a_x[:,None] ,a_y[:,None]), axis=1))
        point.set_offsets(walker_list[i])
    if i == T-1 : line_.set_data(Lightning[:,0], Lightning[:,1])
    return point, agg, line_

anim = animation.FuncAnimation(fig, animate, frames= T+5)
anim.save(filename='1_animation.mp4', writer='ffmpeg', fps=30, codec="png") # png for alpha
plt.close
