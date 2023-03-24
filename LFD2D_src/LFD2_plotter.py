# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 11:01:59 2022

@author: samir
"""

from twoD_Linear_FD import *
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [6.4,4.8]

def LFD2_plotter(f,a,b,c,d,g,n,m):
    
    # Define x and y plotting matrices
    x = np.outer(np.linspace(0,1,n+1), np.ones(n+1))
    y = x.copy().T
    
    # Define true z values from g
    z_true = g(x,y)
    
    # Plot z_true
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.plot_surface(x,y,z_true,cmap='viridis',edgecolor = 'green')
    ax.set_zlim(0,2)
    ax.view_init(45,45)
    plt.show()
    
    # Gather approximation matrix z_pred from twoD_Linear_FD
    z_pred = twoD_Linear_FD(f,a,b,c,d,g,n,m) 
    
    # Print linf err
    print(max(abs(z_true.flatten()-z_pred.flatten())))
        
    # Plot z_pred
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.plot_surface(x,y,z_pred,cmap='viridis',edgecolor = 'green')
    ax.set_zlim(0,2)
    ax.view_init(45,45)
    plt.show()
    
    # Plot the error
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.plot_wireframe(x,y,z_true-z_pred)
    ax.view_init(20,45)
    plt.show()