#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 19:17:22 2022
@author: fabien
"""

#### MODULE
import numpy as np, pylab as plt
import pandas as pd
from itertools import combinations
from matplotlib import animation
    
def elastic_collision(r1, r2, v1, v2, m1, m2):
    # input variable
    M = m1 + m2
    # conservation law
    d = np.linalg.norm(r1 - r2)**2
    u1 = v1 - 2*m2 / M * np.dot(v1-v2, r1-r2) / d * (r1 - r2)
    u2 = v2 - 2*m1 / M * np.dot(v2-v1, r2-r1) / d * (r2 - r1)
    ## return new velocity
    return np.array([u1,u2])

def border_collision(X,V):
    x,y = X
    ### update position
    V[0,x < 0] = -V[0,x < 0]
    V[0,x > 1] = -V[0,x > 1]
    V[1,y < 0] = -V[1,y < 0]
    V[1,y > 1] = -V[1,y > 1]
    return V

################## Particules
s = 5
n = s**2
radii = (0.01*np.random.random(n)+0.01)[None] #radius
grid = ((np.mgrid[0:2*s, 0:2*s]+0.5).reshape(2,-1).T)/(2*s)
loc = np.random.choice(len(grid), n, replace=False)
r = grid[loc].copy().T #position
u = (0.1*np.random.random(n) + 0.05)*np.exp(1j*(2*np.pi * np.random.random(n)))
v = np.column_stack((u.real,u.imag)).T # velocity

Radius = 0.025

#### Iterate
dt = 0.025
T = 500
SOLVE = False

fig = plt.figure(figsize=(5,5))
plt.xlim([0,1]); plt.ylim([0,1])
plot_list = []
for t in range(T) :
    # advance
    r += v * dt
    # border
    border_collision(r,v)
    #r = r % 1
    # collision
    pairs = combinations(range(n), 2)
    for i,j in pairs:
        if np.hypot(*(r[:,i] - r[:,j])) < radii[:,i] + radii[:,j] :
            m1, m2 =  radii[0,i]**2,  radii[0,j]**2
            # direct solution
            v[:,i], v[:,j] = elastic_collision(r[:,i], r[:,j], v[:,i], v[:,j], m1, m2)
    # collective
    u = (v[0]+1j*v[1]).copy() #t-1
    for i in range(len(u)):
        d = np.absolute(u[i] - u) #np.linalg.norm(v[i] - v, axis=1)
        index = np.where((d<Radius))[0]
        # circular mean
        polar_sum = np.sum(np.exp(1j*np.angle(u[index])))
        angle_mean = np.angle(polar_sum)
        #v[:,i] = [dt*np.cos(angle_mean), dt*np.sin(angle_mean)]
    # add field
    v +=  np.array(n* [[0,-0.0005]]).T
    # save plot
    plot_list.append([plt.scatter(r[0],r[1], c='b')])#, s=10000*radii[0])])
    
anim = animation.ArtistAnimation(fig, plot_list, interval=100)
anim.save("animation.mp4", fps=100)