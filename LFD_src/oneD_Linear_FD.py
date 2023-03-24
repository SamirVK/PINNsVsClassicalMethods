# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:29:07 2022

@author: samir
"""

import numpy as np
np.set_printoptions(precision=18, suppress=True)

def oneD_Linear_FD(p,q,r,a,b,alpha,beta,N):
    
    #define the subinterval spacing
    h = (b-a)/(N+1)
    
    #populate A and B in Au=B
    A = np.zeros([N,N])
    B = [0]*N
    
    #populate A
    A[0,0] = 2+h**2*q(a+h)
    A[0,1] = h/2*p(a+h)-1 
    for i in range(1,N-1):
            A[i,i-1] = -1-h/2*p(a+(i+1)*h)
            A[i,i] = 2+h**2*q(a+(i+1)*h)
            A[i,i+1] = h/2*p(a+(i+1)*h)-1
    A[N-1,N-2] = -1-h/2*p(a+N*h)
    A[N-1,N-1] = 2+h**2*q(a+N*h)
    
    #populate B
    B[0] = -h**2*r(a+h)+(1+h/2*p(a+h))*alpha
    for i in range(1,N-1):
        B[i] = -h**2*r(a+(i+1)*h)
    B[N-1] = -h**2*r(a+N*h)+(1-h/2*p(a+N*h))*beta
   
    #linear solve using numpy
    u = np.linalg.solve(A,B)
    u = np.concatenate([[alpha],u,[beta]])
    
    return u