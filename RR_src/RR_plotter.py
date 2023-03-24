# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:56:26 2022

@author: samir
"""

from RR_approx import *
import matplotlib.pyplot as plt
import itertools

def RR_plotter(p,q,f,y,alpha,beta,splines):
    def bigO(h):
        if splines == True:
            return h**4
        return h**2
    hValues = []
    l_infinity_errors = []
    N = 2 
    while (N < 64):
        #define the subinterval lengths
        h = 1/(N+1)
        hValues.append(h)
        #find the true  and approximated function values at the discretization
        #points by calling RR_approx to compute phi for the given N.
        phi = RR_approx(p,q,f,alpha,beta,N,splines)
        u = [0]*N
        u_tilde = [0]*N
        grid_points = [0]*N
        acc_at_grid_points = [0]*N
        for i in range(N):
            u[i] = y((i+1)*h)
            u_tilde[i] = phi((i+1)*h)
            acc_at_grid_points[i] = u[i] - u_tilde[i]
            grid_points[i] = (i+1)*h
        u.insert(0,alpha)
        u.append(beta)
        u_tilde.insert(0,alpha)
        u_tilde.append(beta)
        
        #compute the l_inf norm of the error vector
        absErrors = [0] * (N+2)
        for i in range(N+2):
            absErrors[i] = abs(u[i] - u_tilde[i])
        error = max(absErrors)
        l_infinity_errors.append(error)
        
        #if splines == False:
            #plot the approximator function returned by RR_approx
         #   partition = []
          #  for i in range(0,N+2):
           #     partition.append(i*h)
           # plt.plot(partition, u_tilde)
        #else: 
        x = np.linspace(0,1,100)
        phi_values = []
        target_values = []
        for i in range(100):
            phi_values.append(phi(x[i]))
            target_values.append(y(x[i]))
        accuracy = [e1 - e2 for (e1, e2) in zip(target_values, phi_values)]
        
        ax = plt.gca()
        if N <= 32:
            plt.plot(x,accuracy, label = 'N = %d' % N)
            plt.plot(grid_points,acc_at_grid_points,'.',color = ax.lines[-1].get_color())
            
        N = N * 2
        
    plt.ylabel('Approximation error')
    plt.xlabel('x')
    plt.legend()
    plt.show()
    
    upperBound = list(map(bigO,hValues))
    
    plt.plot(hValues, l_infinity_errors, marker = '.', label = r'max$_i||u_i-\tilde{u}_i||_\infty$')
    if splines == True:
        plt.plot(hValues, upperBound, ':r', label = r'$h^4$')
    else:
        plt.plot(hValues, upperBound, ':r', label = r'$h^2$')
    plt.legend(loc='upper left')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('error')
    plt.xlabel('h')
    plt.show()