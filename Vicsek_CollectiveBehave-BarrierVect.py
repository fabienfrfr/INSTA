#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 16:10:50 2020
@author: fabien
"""
import numpy as np, pylab as plt
import matplotlib.animation as animation
import time

t = time.time()
################## PARAMETER
N_cells, DimX, DimY = 3000, 16, 9
R_fied, strength = 10, 0.15
Radius, step, sigma = 0.5, 0.15, np.pi/10
N_time = 300

################## INIT
cells_state = np.zeros((N_time,N_cells,3))
cells_state[0] = np.random.rand(N_cells,3) #x,y,angles
cells_state[0,:,0] = cells_state[0,:,0] * DimX
cells_state[0,:,1] = cells_state[0,:,1] * DimY
cells_state[0,:,2] = cells_state[0,:,2] * (2*np.pi) - np.pi

################## Barrier circle carterzian equation (arctan quadrant)
X,Y = np.mgrid[0:DimX+1:strength,0:DimY+1:strength]
Angle = np.arctan2(Y-DimY/2, X-DimX/2) + np.pi
F = (X-DimX/2)**2 + (Y-DimY/2)**2 - R_fied
X_,Y_,Angle_ = X[F>0], Y[F>0], Angle[F>0]
false_cells = np.concatenate((X_[:,None],Y_[:,None], Angle_[:,None]), axis=1)

################## TrAJECTORY Calculate
for i in range(N_time-1) :
    for n in range(N_cells) :
        cell_field = np.concatenate((cells_state[i],false_cells))
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

################## PLOT ANIMATION
fig = plt.figure(figsize=(DimX, DimY)) 
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
plt.xticks([]), plt.yticks([])
plt.xlim(0,DimX); plt.ylim(0,DimY)

point = plt.scatter(cells_state[0,:,0], cells_state[0,:,1], color='k', marker = '.', alpha = 0.90, s=10)
#plt.contour(X,Y,F,[0]) #field limit

ax.set_facecolor("None")
fig.patch.set_alpha(0.)
fig.patch.set_facecolor("None")#None for alpha

def animate(i):
    global cells_state, Radius
    point.set_offsets(cells_state[i,:,:2])
    return point,

anim = animation.FuncAnimation(fig, animate, frames= N_time)
anim.save(filename='1_animation.mp4', writer='ffmpeg', fps = 30, codec="png") # png for alpha
plt.close

##################
print(time.time()-t)
