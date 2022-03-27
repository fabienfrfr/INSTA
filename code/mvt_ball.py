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

################## FUNCTION
def border_collision(X,V,r):
    x,y = X
    vx, vy = V
    ### update position
    # x - r < 0:
    x[x - r < 0] = r[x - r < 0]
    vx[x - r < 0] = -vx[x - r < 0]
    # x + r > 1:
    x[x + r > 1] = 1-r[x + r > 1]
    vx[x + r > 1] = - vx[x + r > 1]
    # y - r < 0:
    y[y - r < 0] = r[y - r < 0]
    vy[y - r < 0] = -vy[y - r < 0]
    # y + r > 1:
    y[y + r > 1] = 1-r[y + r > 1]
    vy[y + r > 1] = -vy[y + r > 1]       


def elastic_collision(r1, r2, v1, v2):
    # input variable
    m1, m2 = r1**2, r2**2
    M = m1 + m2
    # conservation law
    d = np.linalg.norm(r1 - r2)**2
    u1 = v1 - 2*m2 / M * np.dot(v1-v2, r1-r2) / d * (r1 - r2)
    u2 = v2 - 2*m1 / M * np.dot(v2-v1, r2-r1) / d * (r2 - r1)
    ## return new velocity
    return (u1,u2)

################## Particules

n = 50
radii = (0.03*np.random.random(n)+0.02)[None] #radius
r = radii + (1 - 2*radii) * np.random.random((2,n)) #position
u = (0.1*np.random.random(n) + 0.05)*np.exp(1j*(2*np.pi * np.random.random(n)))
v = np.column_stack((u.real,u.imag)).T # velocity

#### Iterate
dt = 0.01
T = 1000

fig = plt.figure()
plot_list = []
for t in range(T) :
    # advance
    r += v * dt
    # border
    border_collision(r,v,radii.squeeze())
    # collision
    pairs = combinations(range(n), 2)
    for i,j in pairs:
        if np.hypot(*(r[:,i] - r[:,j])) < radii[:,i] + radii[:,j] :
            m1, m2 = r[:,i]**2, r[:,j]**2
            v[:,i], v[:,j] = elastic_collision(r[:,i], r[:,j], v[:,i], v[:,j])
    # save plot
    plot_list.append([plt.scatter(r[0],r[1])])
    
anim = animation.ArtistAnimation(fig, plot_list, interval=100)
anim.save("animation.mp4")