# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:53:54 2022

@author: samir
"""

import numpy as np
from scipy.integrate import quad
#np.set_printoptions(precision=3, suppress=True)

def RR_approx(p,q,f,alpha,beta,N,splines):
    h = 1/(N+1)
    if (splines == False):
        
        #Define the piecewise-linear basis
        def phi_i(i,x):
            if (x >= 0 and x <= (i-1)*h): 
                return 0
            if (x > (i-1)*h and x <= i*h):
                return (1/h)*(x-(i-1)*h)
            if (x > i*h and x <= (i+1)*h):
                return (1/h)*((i+1)*h-x)
            return 0
        
        #Define 6 integrator functions using piecewise-linear interpolations
        def Q_1(i):
            return (h/12) * (q(i*h) + q((i+1)*h))
        def Q_2(i):
            return (h/12) * (3*q(i*h) + q((i-1)*h))
        def Q_3(i):
            return (h/12) * (3*q(i*h) + q((i+1)*h))
        def Q_4(i):
            return (1/(2*h)) * (p(i*h) + p((i-1)*h))
        def Q_5(i):
            return (h/6) * (2*f(i*h) + f((i-1)*h))
        def Q_6(i):
            return (h/6) * (2*f(i*h) + f((i+1)*h))
        
        #populate A and B in Ac=B
        A = np.zeros([N,N])
        B = [0]*N
        
        for i in range(N):
            for j in range(N):
                if i == j: 
                    A[i,j] = Q_4(i+1) + Q_4(i+2) + Q_2(i+1) + Q_3(i+1)
                if i + 1 == j: 
                    A[i,j] = -Q_4(i+2) + Q_1(i+1)
                if i - 1 == j:
                    A[i,j] = -Q_4(i+1) + Q_1(i)
        
        for i in range(N):
            B[i] = Q_5(i+1) + Q_6(i+1)
                    
        c = np.linalg.solve(A,B)
        
        #Define the approximator function phi
        def phi(x):
            u_tilde = 0
            for i in range(N):
                u_tilde = u_tilde + c[i] * phi_i(i+1, x)
            return u_tilde
        return phi
    else:
        
        #Define the basic spline function S and its derivative S'
        def S(x):
            if (x <= -2):
                return 0
            if (x >= -2 and x <= -1):
                return (1/4)*(2+x)**3
            if (x > -1 and x <= 0):
                return (1/4)*((2+x)**3 - 4*(1+x)**3)
            if (x > 0 and x <= 1):
                return (1/4)*((2-x)**3 - 4*(1-x)**3)
            if (x > 1 and x <= 2):
                return (1/4)*(2-x)**3
            return 0
        
        def S_prime(x):
            if (x <= -2):
                return 0
            if (x > -2 and x <= -1):
                return (3/4)*(2+x)**2
            if (x > -1 and x <= 0):
                return (3/4)*(2+x)**2 - 3*(1+x)**2
            if (x > 0 and x <= 1):
                return -(3/4)*(2-x)**2 + 3*(1-x)**2
            if (x > 1 and x <= 2):
                return -(3/4)*(2-x)**2
            return 0

        #Define the basis function and its derivative
        def phi_i(i,x):
            if (i == 0):
                return S(x/h) - 4*S((x+h)/h)
            if (i == 1):
                return S((x-h)/h) - S((x+h)/h)
            if (i >= 2 and i <= N-1):
                return S((x-i*h)/h)
            if (i == N):
                return S((x-N*h)/h) - S((x-(N+2)*h)/h)
            if (i == N+1):
                return S((x-(N+1)*h)/h) - 4*S((x-(N+2)*h)/h)
        
        def phi_i_prime(i,x):
            if (i == 0):
                return (1/h)*S_prime(x/h) - (4/h)*S_prime((x+h)/h)
            if (i == 1):
                return (1/h)*S_prime((x-h)/h) - (1/h)*S_prime((x+h)/h)
            if (i >= 2 and i <= N-1):
                return (1/h)*S_prime((x-i*h)/h)
            if (i == N):
                return (1/h)*S_prime((x-N*h)/h) - (1/h)*S_prime((x-(N+2)*h)/h)
            if (i == N+1):
                return (1/h)*S_prime((x-(N+1)*h)/h) - (4/h)*S_prime((x-(N+2)*h)/h)
        
        #Define the integrands
        def integrand_a(x,i,j):
            return p(x)*phi_i_prime(i,x)*phi_i_prime(j,x)+q(x)*phi_i(i,x)*phi_i(j,x)
        def integrand_b(x,i):
            return f(x)*phi_i(i,x)
        
        #populate A and B in Ac=B
        A = np.zeros([N+2,N+2])
        B = [0]*(N+2)

        for i in range(N+2):
            for j in range(min(i+4,N+2)):
                if j >= i:
                    A[i,j] = quad(integrand_a, max((i-2)*h,0), min((j+2)*h,1), args = (i,j))[0]
                else:
                    A[i,j] = A[j,i]
        
        for i in range(N+2):
            B[i] = quad(integrand_b, max((i-2)*h,0), min((i+2)*h,1), args = (i))[0]
        
        c = np.linalg.solve(A,B)
        
        #Define the approximator function phi
        def phi(x):
            u_tilde = 0
            for i in range(N+2):
                u_tilde = u_tilde + c[i] * phi_i(i, x)
            return u_tilde
        return phi









    