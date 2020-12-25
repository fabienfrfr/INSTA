#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 19:29:31 2020
@author: fabien
"""

import numpy as np, pylab as plt
import matplotlib.animation as animation
import time
from scipy.optimize import fsolve

################## FUNCTION
def central_axial_symmetry_coor(V, *data):
    a, b, x_prev, y_prev = data
    f1 = b*V[0] - a*V[1] + b*x_prev - a*y_prev # Thales
    f2 = V[0]**2 + V[1]**2 - (x_prev**2 + y_prev**2) # Pythagore
    return [f1,f2]

t = time.time()
################## PARAMETER
N_cells, DimX, DimY = 500, 16, 9
Radius, step, sigma = 0.5, 0.15, np.pi/10 #vicsek
R_fied, strength = 10, 3.25 #barrier
R_coll, v_pred = 0.25, 0.25 #one predator
N_time = 300

################## INIT
# boids cells
cells_state = np.zeros((N_time,N_cells,3))
cells_state[0] = np.random.rand(N_cells,3) #x,y,angles
cells_state[0,:,0] = cells_state[0,:,0] * DimX
cells_state[0,:,1] = cells_state[0,:,1] * DimY
cells_state[0,:,2] = cells_state[0,:,2] * (2*np.pi) - np.pi

# predator
preds = np.zeros((N_time,3))
preds[0] = np.random.rand(3) #x,y,angles
preds[0] = [preds[0,0] * DimX, preds[0,1] * DimY, preds[0,2] * (2*np.pi) - np.pi]

################## Barrier circle carterzian equation (arctan quadrant)
X,Y = np.mgrid[0:DimX+1:strength,0:DimY+1:strength]
Angle = np.arctan2(Y-DimY/2, X-DimX/2) + np.pi
F = (X-DimX/2)**2 + (Y-DimY/2)**2 - R_fied
X_,Y_,Angle_ = X[F>0], Y[F>0], Angle[F>0]
false_cells = np.concatenate((X_[:,None],Y_[:,None], Angle_[:,None]), axis=1)


################## TrAJECTORY Calculate
cell_range = np.arange(N_cells)
for i in range(N_time-1) :
    # One Predator pseudo-collision
    # ADD Quadtree Collision for more than ONE !!
    d = np.linalg.norm(cells_state[i,:,:2] - preds[i,:2], axis=1)
    indexes = np.where((d<R_coll))[0]
    if indexes.size == 0 :
        v = v_pred*np.cos(preds[i,2]), v_pred*np.sin(preds[i,2])
        preds[i+1] = np.mod(preds[i,0] + v[0], DimX), np.mod(preds[i,1] + v[1], DimY), preds[i,2]
    else :
        orthgnl = np.concatenate((preds[i,None],cells_state[i,indexes]), axis=0)
        bissectrc = np.sum(np.exp(1j*orthgnl[:,2]))
        angle_new = []
        for o in orthgnl :
            p_vect = v_pred*np.cos(o[2]), v_pred*np.sin(o[2])
            #axial transformation (through to 0)
            ab = bissectrc.real, bissectrc.imag
            root = fsolve(central_axial_symmetry_coor, [1,1], args=(ab[0], ab[1], p_vect[0], p_vect[1]))
            angle_new += [np.angle(root[0]+1j*root[1])]
        angle_new = np.array(angle_new)
        # predator collision
        v_ = v_pred*np.cos(angle_new[0]), v_pred*np.sin(angle_new[0])
        preds[i+1] = np.mod(preds[i,0] + v_[0], DimX), np.mod(preds[i,1] + v_[1], DimY), angle_new[0]
        # preys collision
        v = step*np.cos(angle_new[1:]), step*np.sin(angle_new[1:])
        cells_state[i+1,indexes, 0] = np.mod(cells_state[i,indexes,0] + v[0], DimX)
        cells_state[i+1,indexes, 1] = np.mod(cells_state[i,indexes,1] + v[1], DimY)
        cells_state[i+1,indexes, 2] = angle_new[1:]
    # Vicsek (for no collision boids)
    cell_field = np.concatenate((cells_state[i],false_cells))
    for n in np.delete(cell_range, indexes) :
        # euclidean distance (with CLP modulus)
        d = np.linalg.norm(cell_field[:,:2] - cells_state[i,n,:2], axis=1)
        index = np.where((d<Radius))[0]
        # circular mean (example : mean(350,0) = 355)
        polar_sum = np.sum(np.exp(1j*cell_field[index,2]))
        angle_mean = np.angle(polar_sum)
        # normal noise
        angle_rand = np.random.normal(angle_mean, sigma, 1)
        v = step*np.cos(angle_rand), step*np.sin(angle_rand)
        # change position (with CLP modulus)
        cells_state[i+1,n] = np.mod(cells_state[i,n,0] + v[0], DimX), np.mod(cells_state[i,n,1] + v[1], DimY), angle_mean
    print(i)

################## PLOT ANIMATION
fig = plt.figure(figsize=(DimX, DimY)) 
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
plt.xticks([]), plt.yticks([])
plt.xlim(0,DimX); plt.ylim(0,DimY)

point = plt.scatter(cells_state[0,:,0], cells_state[0,:,1], color='k', marker = '.', alpha = 0.90, s=10)
line_pred = ax.plot([],[], color='k')[0]
#plt.contour(X,Y,F,[0]) #field limit
#plt.streamplot(Y, X, F, Angle) #streamline

ax.set_facecolor("white") #None
fig.patch.set_alpha(0.)
fig.patch.set_facecolor("None")#None for alpha

def animate(i):
    point.set_offsets(cells_state[i,:,:2])
    line_pred.set_data(preds[i-1:i+1,0], preds[i-1:i+1,1])
    return point, line_pred

anim = animation.FuncAnimation(fig, animate, frames= N_time)
anim.save(filename='1_animation.mp4', writer='ffmpeg', fps = 30, codec="png") # png for alpha
plt.close

##################
print(time.time()-t)
