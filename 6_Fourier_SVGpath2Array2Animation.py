#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 12:07:59 2020
@author: fabien
Only available with path (no circle for exemple)
"""

import pylab as plt, numpy as np
from xml.dom import minidom
import svg.path as svg
from scipy.integrate import simps
import matplotlib.animation as animation

################## MANUAL PARAMETER
N_series = 100
n_bezier_segment = 5

################## Cubic bezier to arrays FUNCTION
def cubic_bezier_sample(start, control1, control2, end):
    inputs = np.array([start, control1, control2, end])
    cubic_bezier_matrix = np.array([
        [-1,  3, -3,  1],
        [ 3, -6,  3,  0],
        [-3,  3,  0,  0],
        [ 1,  0,  0,  0]
    ])
    partial = cubic_bezier_matrix.dot(inputs)
    return (lambda t: np.array([t**3, t**2, t, 1]).dot(partial))

################## EXTRACT PATH and PARAM of SVGfile
doc = minidom.parse("1_BabFes.svg")
svg_info = doc.getElementsByTagName('svg')
h,w = int(svg_info[0].getAttribute('height')[:4]), int(svg_info[0].getAttribute('width')[:4])
path_d = np.array([p.getAttribute('d') for p in doc.getElementsByTagName('path')])
path_style = [p.getAttribute('style') for p in doc.getElementsByTagName('path')]
path_fill = np.array([p.getAttribute('fill') for p in doc.getElementsByTagName('path')])
doc.unlink()

if True :
    color = [p.split('ill:')[1].split(';')[0] for p in path_style]
else :
    index_color = np.where(['#' in f for f in path_fill])[0]
    color = path_fill[index_color]
    path_d = path_d[index_color]

################## PATH to VECTOR and Distance
coor_p, dist_p = [],[]
T = []
for p in path_d :
    #extract coordinate
    path = svg.parse_path(p) 
    v = []
    for e in path :
        if isinstance(e, svg.path.Line) or isinstance(e, svg.path.Close):
            v += [e.start]
        elif isinstance(e, svg.path.CubicBezier) :
            start = np.array([e.start.real, e.start.real])
            control1 = np.array([e.control1.real, e.control1.imag])
            control2 = np.array([e.control2.real, e.control2.imag])
            end = np.array([e.end.real, e.end.imag])
            curve = cubic_bezier_sample(start, control1, control2, end)
            points = np.array([curve(t) for t in np.linspace(0, 1, n_bezier_segment)])
    v0 = np.array(v + [e.end])
    v1 = np.roll(v0,1)
    # Euclidean dist sum
    dist = np.sqrt((v1.real-v0.real)**2+(v1.imag-v0.imag)**2)
    d, i = np.zeros(dist.shape), 0
    for d_ in dist :
        if i == 0 : d[i] = d_
        else : d[i] = d[i-1] + d_
        i+=1
    coor_p += [v0]
    dist_p += [d]
    T += [d.max()]

################## SAMPLING INTERPOLATION OF Vector (for FS)
N = int(max(T))
dist_interp, vector_coor = [],[]
for d_, v_ in zip(dist_p, coor_p) :
    dist_interp += [np.linspace(0,d_.max(),2*N)]
    vector_coor += [[np.interp(dist_interp[-1], d_, v_.real) , np.interp(dist_interp[-1], d_, v_.imag)]]
    #joint excatly 2 edding points (discontinuity otherwise)
    vector_coor[-1][0][0], vector_coor[-1][0][-1] = v_.real[0], v_.real[-1]
    vector_coor[-1][1][0], vector_coor[-1][1][-1] = v_.imag[0], v_.imag[-1]
    dist_interp[-1][-1] = d_.max()
    
################## XY FOURIER COEFF for each shape
Cf = []
for v_,d_, T_ in zip(vector_coor, dist_interp, T) :
    cf = []
    for n in range(N_series):
        cf += [[simps(v_[0]*np.exp(-1j*2*n*np.pi*d_/T_)/T_, d_), simps(v_[1]*np.exp(-1j*2*n*np.pi*d_/T_)/T_, d_)]]
    Cf += [cf]
Cf = np.array(Cf)

################## FOURIER SERIES for each shape
Nb_path = len(path_d)
SERIES_X = np.zeros((N_series,Nb_path,2*N)).astype('complex128')
SERIES_Y = SERIES_X.copy()
for n in range(Nb_path) : 
    SERIES_X[0,n,:] = Cf[n,0,0]
    SERIES_Y[0,n,:] = Cf[n,0,1]
for k in range(1,N_series) :
    for n in range(Nb_path) :
        #harmonique C_n = C_-n
        Hx = 2*Cf[n,k,0]*np.exp(1j*2*k*np.pi*dist_interp[n]/T[n]) 
        Hy = 2*Cf[n,k,1]*np.exp(1j*2*k*np.pi*dist_interp[n]/T[n])
        #sum S + H
        SERIES_X[k,n,:] = SERIES_X[k-1,n,:] + Hx
        SERIES_Y[k,n,:] = SERIES_Y[k-1,n,:] + Hy

################## ANIMATE FOURIER SERIES
fig = plt.figure(figsize=(w/100, h/100), dpi=360) 
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
plt.xticks([]), plt.yticks([])
plt.xlim(0,w); plt.ylim(0,h)

fill_list = [ax.fill(SERIES_X[1,n,:].real,SERIES_Y[1,n,:].real, color[n], animated=True) for n in range(Nb_path)]

def animate(i):
    for n in range(Nb_path) :
        #create new array
        xy = np.array([SERIES_X[i+1,n,:].real,SERIES_Y[i+1,n,:].real])
        # change polygon
        fill_list[n][0].set_xy(xy.T)
    return fill_list

anim = animation.FuncAnimation(fig, animate, frames= N_series-1)
anim.save(filename= '2_Animation.mp4', writer='ffmpeg', fps=20)
plt.close
