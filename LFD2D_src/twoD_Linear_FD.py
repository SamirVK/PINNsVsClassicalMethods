# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 09:42:28 2022

@author: samir
"""

import numpy as np
np.set_printoptions(precision=18, suppress=True)

def twoD_Linear_FD(f,a,b,c,d,g,n,m):
    
    # Define the subinterval spacing
    h = (b-a)/n
    k = (d-c)/m
    
    # Create A and B in Aw = B
    A = np.zeros([(n-1)*(m-1),(n-1)*(m-1)])
    B = [0]*((n-1)*(m-1))
    
    for i in range(1,n):
        for j in range(1,m):
            # Define l, x_i, and y_j
            l = i + (m - 1 - j)*(n - 1)
            x = a + i*h
            y = c + j*k
            
            # Evaluate f
            B[l-1] = -h**2*f(x,y)
            
            # Evaluate each of the 5 adjacent mesh points
            # (i,j)
            A[l-1,l-1] = 2*((h/k)**2 + 1)
            
            # (i+1,j)
            if (i+1) == n:
                B[l-1] += g(b,y)
            else:
                A[l-1,l] = -1
                
            # (i-1,j)
            if (i-1) == 0: 
                B[l-1] += g(a,y)
            else: 
                A[l-1,l-2] = -1
                
            # (i,j+1)
            if (j+1) == m:
                B[l-1] += g(x,d)
            else:
                A[l-1,(l-1) - (n-1)] = -(h/k)**2
            
            # (i,j-1)
            if (j-1) == 0:
                B[l-1] += g(x,c)
            else:
                A[l-1,(l-1) + (n-1)] = -(h/k)**2
            
    # Linear solve using numpy
    w = np.linalg.solve(A,B)
        
    # Return a 2D matrix of approximations w_l
    W = np.zeros([n+1,m+1])
    for i in range(n+1):
        for j in range(m+1):
            if i == 0 or j == 0 or i == n or j == m:
                x = a + i*h
                y = c + j*k
                W[i,j] = g(x,y)
            else:
                l = i + (m - 1 - j)*(n - 1)
                W[i,j] = w[l-1]
        
    return W