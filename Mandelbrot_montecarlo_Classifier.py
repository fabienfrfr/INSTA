#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:17:54 2020
@author: fabien
https://python-prepa.github.io/stochastique.html
https://scipython.com/book/chapter-7-matplotlib/problems/p72/the-julia-set/
"""

import numpy as np, pylab as plt
import matplotlib.animation as animation
from sklearn.neighbors import KNeighborsClassifier

############################################### FUNCTION
def julia(z):
    #return z**2 + complex(-3./4,0)
    return z**2 + complex(0,1)

############################################### PARAMETER
Nmax, Zmax = 10, 10
N_point, t_ani = 10000, 10
frame_sec = int(N_point/t_ani)
Lmin, L_max = -1.5, 1.5

dx = 0.02 #step mesh prediction
a = 100 #animation factor (point/frame)

############################################### RANDOMISED JuliaMap
# Complex numbers
c_ab = np.random.uniform(Lmin, L_max,(N_point,2))
x,y = c_ab.T 

# Iteration : z**2 + i
values = []
for c in c_ab :
    # initialization
    z = complex(c[0],c[1])
    n = 0
    while n < Nmax and abs(z) < Zmax :
        z = julia(z)
        n+=1
    if abs(z) < 1 : n = 0
    values += [n]
values = np.array(values)

# Monte-carlo verification
S_fract = len(values[values == 0])/len(values[values > 0])
print(S_fract)

############################################### CLASSIFICATION
# construct supervised target
target = values.copy()
target[(target > 0)*(target < Nmax)] = 2
target[target >= Nmax] = 1

# fitting
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(c_ab, target)

# prediction & contouring
xx, yy = np.meshgrid(np.arange(Lmin, L_max, dx), np.arange(Lmin, L_max, dx))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, Z); plt.savefig('2_Dendrite_julia.svg'); plt.close()

############################################### ANIMATION SCATTERING
# x,y reconstruction (init outside lim scattering)
x_, y_ = x+10, y+10

fig = plt.figure(figsize=(5,5), dpi=120) 
fig.add_axes([0., 0., 1., 1.], frameon=False)
plt.xticks([]), plt.yticks([])
plt.xlim(-1.25,1.25); plt.ylim(-1.25,1.25)

# set figure background opacity (alpha) to 0
fig.patch.set_alpha(0.)

# Création des noeud/arete à mettre à jour :
point = plt.scatter(x_, y_, c=values, marker = '.', s=10)
fig.patch.set_facecolor("None")#None for alpha

def animate(i):
    END = int(i*a)
    x_[:END], y_[:END] = x[:END], y[:END]
    x_[values < 8], y_[values < 8] = 10, 10
    point.set_offsets(np.concatenate((x_[:,None] ,y_[:,None]), axis=1))
    if i%100 == 0 : print(i)
    return point,

anim = animation.FuncAnimation(fig, animate, frames= int(N_point/a))
anim.save(filename='1_animation.mp4', writer='ffmpeg', fps=30, codec="png") # png for alpha
plt.close
