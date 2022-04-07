#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 11:31:25 2022

@author: fabien
"""
import numpy as np
import scipy.sparse as sp


L=3
H=4
T=0.1
N=20
M=15
l=L/(N+1)
h=H/(M+1)
dt=0.5*h
x=np.linspace(l,L-l,N)
y=np.linspace(h,H-h,M)
from mpl_toolkits import mplot3d
X, Y = np.meshgrid(np.linspace(0,L,N+2),np.linspace(0,H,M+2))
#construction de la matrice en systeme creux
D0=(1/dt+2/(l**2)+2/(h**2))*np.ones(N*M)# diagonale principale
D1=-1/l**2*np.ones(N*M)# surdiagonale
D1[N::N]=0.#correction de la surdiagonale (voisin de droite n existe pas au bord droit)
DM1=-1/l**2*np.ones(N*M)# sousdiagonale
DM1[N-1::N]=0.#correction de la sousdiagonale (voisin de gauche n existe pas au bord gauche)
DN=-1/h**2*np.ones(N*M)
A=sp.spdiags(D0,[0],N*M,N*M)+sp.spdiags(D1,[1],N*M,N*M)+sp.spdiags(DM1,[-1],N*M,N*M)
A=A+sp.spdiags(DN,[N],N*M,N*M)+sp.spdiags(DN,[-N],N*M,N*M)