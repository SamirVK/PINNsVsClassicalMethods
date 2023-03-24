# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:09:12 2022

@author: samir
"""

from oneD_nonLinear_FD import *
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [6.4,4.8]

def bigO(h):
    return h**2

def NLFD_plotter(f,f_y,f_y_prime,y,a,b,alpha,beta):
    hValues = []
    l_infinity_errors = []
    N = 8  
    while (N < 1024):
        #define the subinterval lengths
        h = (b-a)/(N+1)
        hValues.append(h)
        #find the true function values at the discretization points
        u = [0]*N
        for i in range(1,N+1):
            u[i-1] = y(a+i*h)
        u.insert(0,alpha)
        u.append(beta)
        #call LFD to compute approximations at discretization points
        u_tilde = oneD_nonLinear_FD(f,f_y,f_y_prime,a,b,alpha,beta,N)
        
        #compute the l_inf norm of the error vector
        error = max(abs(u - u_tilde))
        l_infinity_errors.append(error)
        
        partition = []
        for i in range(0,N+2):
            partition.append(a+i*h)
        accuracy = [e1 - e2 for (e1, e2) in zip(u, u_tilde)]
        if N <= 128:
            plt.plot(partition, accuracy, marker = '.', label = 'N = %d'% N)
        
        N = N * 2
    
    plt.ylabel('Approximation error')
    plt.xlabel('x')
    plt.legend()
    plt.show()
    
    upperBound = list(map(bigO,hValues))
    
    plt.plot(hValues, l_infinity_errors, marker='.', label = r'max$_i||u_i-\tilde{u}_i||_\infty$')
    plt.plot(hValues, upperBound, ':r', label = r'$h^2$')
    plt.legend(loc='upper left')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('error')
    plt.xlabel('h')
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    