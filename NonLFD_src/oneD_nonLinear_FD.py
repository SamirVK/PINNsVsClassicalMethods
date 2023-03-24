# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:08:05 2022

@author: samir
"""

import numpy as np
np.set_printoptions(precision=18, suppress=True)

def oneD_nonLinear_FD(f,f_y,f_y_prime,a,b,alpha,beta,N):
    #define the subinterval spacing
    h = (b-a)/(N+1)
    
    #initial approximation
    w = [0]*N
    for i in range(1,N+1):
        w[i-1] = alpha + i*((beta-alpha)/(b-a))*h
    w.insert(0,alpha)
    w.append(beta)
    
    k = 1
    while(k < 10):
        #create empty Jacobian and B in Jv=B
        J = np.zeros([N,N])
        B = [0]*N
    
        #populate J
        for i in range(N):
            for j in range(N):
                if (i == j - 1):
                    J[i,j] = -1 + (h/2)*f_y_prime(a+(i+1)*h,w[i+1],(w[i+2]-w[i])/(2*h))
                if (i == j):
                    J[i,j] = 2 + h**2*f_y(a+(i+1)*h,w[i+1],(w[i+2]-w[i])/(2*h))
                if (i == j + 1):
                    J[i,j] = -1 - (h/2)*f_y_prime(a+(i+1)*h,w[i+1],(w[i+2]-w[i])/(2*h))

        #populate B
        for i in range(N):
            B[i] = -(-w[i]+2*w[i+1]-w[i+2]+h**2*f(a+(i+1)*h,w[i+1],(w[i+2]-w[i])/(2*h)))
        
        #prepare w[i] for linear solve
        w = np.delete(w,[0,N+1])
   
        #linear solve using numpy
        v = np.linalg.solve(J,B)
        w = w + v
        
        #reshape w before continuing
        w = np.concatenate([[alpha],w,[beta]])
        
        if(np.linalg.norm(v) < 10**-8):
            break
        k = k + 1
       
    return w