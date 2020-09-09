#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:43:23 2020
@author: fabien
https://python-prepa.github.io/edp_chaleur.html
http://hplgit.github.io/INF5620/doc/pub/sphinx-diffu/._main_diffu000.html
http://galusins.univ-tln.fr/ENSEIGN/Chaleur_Differencefinie-Correc.html
https://docs.scipy.org/doc/scipy/reference/tutorial/linalg.html
"""

import numpy as np, pylab as plt
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp
from skimage import filters as ft
import cv2
from matplotlib import colors

############################################### PARAMETER
L, H, N, T = 5, 5, 140, 10 # Largeur, Hauteur
dx, dy = L/(N+1), H/(N+1), 
e_, dt = 0.001, 0.25

Du, Dv, F, k = 0.28, 0.12, 0.035, 0.065 # Grayscott

color1, color2 = "B69756", "130e0b"

############################################### LAPLACIAN CONSTRUCT
########### MATRIX METHODS (spsolve)
# Definition of the 1D Lalace operator (for L)
lower, main, upper = -1/dx**2, (1/e_+2/(dx**2)+2/(dy**2))/2, -1/dx**2
data = [lower*np.ones(N),main*np.ones(N),upper*np.ones(N)]   # Diagonal terms
offsets = np.array([-1,0,1])                   # Their positions
LAP = sp.dia_matrix( (data,offsets), shape=(N,N)) #print LAP.todense(); plt.spy(LAP)

NN = N*N
I1D = sp.eye(N,N)  

# 2D Laplace operator (tensor product : A*B = (a0*B ... aN*B))
LAP2 = sp.kron(LAP,I1D)+sp.kron(I1D,LAP)

# Asymetric correction (L != H)
DN=((1/dx**2)-1/dy**2)*np.ones(N*N)
LAP2=LAP2+sp.spdiags(DN,[N],N**2,N**2)+sp.spdiags(DN,[-N],N**2,N**2)

########### FINITE METHODS (convolution)
kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]],np.float32)*dt
"""
OTHERS METHODs (ideas) :
Lu = (Utab[0:-2,1:-1] + Utab[1:-1,0:-2] - 4*Utab[1:-1,1:-1] + Utab[1:-1,2:] + Utab[2:  ,1:-1])/(dx**2)
Fu, Fk = np.fft.fft(U), np.fft.fft(kernel) ; Lu = np.fft.ifft(Fu*Fk)
"""
############################################### C. I.
Z = np.zeros((N+2,N+2), [('U', np.double), ('V', np.double)])
U,V = Z['U'], Z['V']
u,v = U[1:-1,1:-1], V[1:-1,1:-1] #pointer parenting
U[...] = 1.0

# Local Noise
r,n = 10, 3
for i in np.random.randint(N-2*r, size=(n,2)):
    u[i[0]-r:i[0]+r,i[1]-r:i[1]+r] = 0.50
    v[i[0]-r:i[0]+r,i[1]-r:i[1]+r] = 0.25

# Front attenuation
U[1:-1,1:-1], V[1:-1,1:-1] = ft.gaussian(u,1), ft.gaussian(v,1)

############################################### PLOT DECLARATION
cmap_ = colors.ListedColormap(['#'+color1, '#'+color2])
bounds=[0,0.5,1.0]
norm = colors.BoundaryNorm(bounds, cmap_.N)

plt.ion()
size = np.array(Z.shape)
dpi = 128 #72.0
figsize= size[1]/float(dpi),size[0]/float(dpi)
fig = plt.figure(figsize=(5,5), dpi=dpi, facecolor="white")
fig.add_axes([0.0, 0.0, 1.0, 1.0], frameon=False)
im = plt.imshow(V[5:-5,5:-5], interpolation='gaussian')#, cmap=cmap_)
plt.xticks([]), plt.yticks([])

############################################### TIME LOOP RESOLUTION
# Log scalling
l, T, N_scale = 2, 100, 10
M = np.repeat(l**(np.arange(1,T/N_scale + 1)), N_scale) #multiplicity
seq = [0] # init sequences progression
for m in M : seq += [seq[-1] + m]
seq = np.array(seq, np.int)

n = 0
for i in range(seq[-1]+1) :
    ### Laplacian
    if i % seq[n] == 0 :
        #solver [S = A-¹ * Vb ; A-¹ = (1/det(A))*adj(A) such A*A-¹=In]:
        uu, vv = u.reshape(N**2), v.reshape(N**2)
        Lu_, Lv_ = spsolve(LAP2, uu/e_) - uu, spsolve(LAP2, vv/e_) - vv
        Lu_, Lv_ = Lu_.reshape((N,N)), Lv_.reshape((N,N))
        #conv :
        LU, LV = cv2.filter2D(U,-1,kernel), cv2.filter2D(V,-1,kernel)
        #mean (compromise between stability and border effect) :
        Lu, Lv = (Lu_ + LU[1:-1,1:-1])/2, (Lv_ + LV[1:-1,1:-1])/2
    else :
        Lu, Lv = cv2.filter2D(U,-1,kernel)[1:-1,1:-1], cv2.filter2D(V,-1,kernel)[1:-1,1:-1]
    
    ### Finite methods resolution
    uvv = u*v*v
    u += (Du*Lu - uvv +  F   *(1-u))
    v += (Dv*Lv + uvv - (F+k)*v    )
    
    ### Save image
    if i % seq[n] == 0 :
        im.set_data(U[5:-5,5:-5])
        im.set_clim(vmin=U.min(), vmax=U.max())
        plt.draw()
        # To make movie
        plt.savefig("DATA/%03d.png" % n ,dpi=dpi)
        n += 1
        print(i)
