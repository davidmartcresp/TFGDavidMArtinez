# -*- coding: utf-8 -*-
"""
@author: David Martínez Crespo
"""
from numpy import *
import sympy as sp
from scipy.optimize import newton_krylov,anderson
import time
class rk:
    """
    An instance of this class will represent an integrator that implements a Runge-Kutta method. 
    The Runge-Kutta method is specified by it's Butcher's table 
          c | A
          --------
            | b^T
     By default the instance will be an explicit Runge Kutta of 4th order represented by the Butcher't table
      
           0  | 0     0    0    0
          1/2 | 1/2   0    0    0
          1/2 | 0     1/2  0    0
           1  | 0     0    1/2  0
          -----------------------
              | 1/6   1/3  1/3  1/6
            
    
    ATRIBUTES:
        ''A'': -- numpy array (default value = array([[0,0,0,0],[0.5,0,0,0],[0,0.5,0,0],[0,0,0.5,0]])); Represents the
                    A matrix of the Butcher's Table.
        ''b'': -- numpy array (default value = array([1./6.,1./3.,1./3.,1./6.]); Represents the
                    b vector of the Butcher's Table.
        ''c'':-- numpy array (default value = array([0,0.5,0.5,1.]); Represents the
                    c vector of the Butcher's Table.
        ''explic'':-- bool  (default value= True); True if the Runge-Kutta method is explicit.
    METHODS:
        explicit(self):  Returns True if self is instance of an explicit Runge-Kutta and 
                            sets the value of  the variable self.explic
        __call__(self,rhs,u0,N=500,t0=0.,tf=1.) OUTPUT: '''rest,y'': --performs the ineration of the system given  by rhs 
                        the initial condition is u0, N the number of subintervals an t0, tf the initial and final time.
  
    NOTES: (README)
        -- rhs functions should depend on time varaibles:
            An allowed function will be
            
             rhs(t,x1,...,xn)=array([y1,...,yn])
            
        --Explicit an implicit methods are treated separatedly, implicit methods performs the solution of the nonlinerasystem with the
           numpy's newton_krylov method.
        -- If the convergence of the Newton's method fails the result alredy calculated will be returned with a message of the error
    WARNING:
        -- Exception treatment is not yet supported, even if the integration is not yet finished when some problem arises any exception
            will be raised.
        -- The accuracy of the butcher's table given by the user is not checked at all even the data types.
    TODO:
        -- Implement the managment of exceptions in the constructor.
    EXAMPLE:
        #from HamiltonianoDePontriagin.py we will take the sist variable.
        #execute after HamiltonianoDePontriagin.py
        
        A=np.array([[0.25,0.25-math.sqrt(3)/6],[0.25+math.sqrt(3)/6,0.25]])
        b=np.array([0.5,0.5])
        c=np.array([0.5-math.sqrt(3)/6,0.5+math.sqrt(3)/6])
        
        sistevaluable=lambdify([t]+X+P+co,Matrix(sist))
        rkGL=rk(A,b,c)
        
        u0=np.array([1, 1, 1, 0, 1, 0, 0, 0, 1.19383, -1.06154, 10.7773, -0.46899, -2.95481, -0.381508, 0, -0.314177])
        te,Y=rkGL(sistevaluable,u0,tf=5,N=1000)
    """
    def __init__(self,A=array([[0,0,0,0],[0.5,0,0,0],[0,0.5,0,0],[0,0,0.5,0]]),b=array([1./6.,1./3.,1./3.,1./6.]),c=array([0,0.5,0.5,1.]) ):
        self.A=A
        self.b=b
        self.c=c
        self.explicit()
        
    def explicit(self):
        """
        Returns True if self is instance of an explicit Runge-Kutta and 
        sets the value of  the variable self.explic
        """
        self.explic=True
        for i in range(len(self.A)):
            for j in range(i,len(self.A[i])):
                if self.A[i][j] !=0:
                    self.explic=False
        return self.explic
                    
    def __call__(self,rhs,u0,N=500,t0=0.,tf=1.):
        """
            INPUT:
               ''rhs''-- function; Function that depends on n+1 variables and return an n-simensional 
                           numpy array:
                               rhs(t,x1,...,xn)=array([y1,...,yn])
              ''u0''-- numpy array; n-dimensional array that represents the initial condition.
              ''N'' -- int; number of subintervals.
              ''t0''-- double; initial time
              ''tf''-- double; final time
            OUTPUT:
                ''rest'':-- list of times
                ''y'':--    list of numpy array's with the point of the solution at the time represented by rest.
        """
        h=(tf-t0)/N
        rest=[t0]
   
        y=[u0]
        t=t0
        if self.explic:
            for k in range(N):
                
                #cálculo de las k's
                kas=[]
                yaux=y[k] #cálculo de k+1 para el caso explícito
                for i in range(len(self.b)):
                    for j in range(i-1):
                        yaux=[yaux[m]+h*self.A[i][j]*kas[j][m][0] for m in range(len(yaux))]
                    kas.append(rhs(t+h*self.c[i],*array(yaux)))

                
                #cálculo de y k+1
                yaux=y[k]
                
                for i in range(len(self.b)):
                    yaux=[yaux[m]+h*self.b[i]*kas[i][m][0] for m in range(len(yaux))]

                
                y.append(array(yaux))
                t+=h
                rest.append(t)
        else: #para el caso implícito
            tiempo=time.time()
            #hay que resolver un sistema de ecuaciones no lineales para cada iteración, para ello generamos 
            # las variables en forma de array y en forma de lista 
            kasymbolic=[sp.symbols('k_%d_0:%d' %(r,len(u0))) for r in range(len(self.b))]
            kasymlist=[r for m in kasymbolic for r in m ]
            
            #Para cada paso de la iteración
            guess = zeros_like(kasymlist) #suposición inicial del método de newton
            for k in range(N):
                
                def sistarray(P):
                    
                    kas= [] # las ecuaciones de las k's se guardan aquí
                     #cálculo de k+1
                    l=0
                    for i in range(len(self.b)):
                        yaux=y[k]
                        for j in range(len(self.b)):
                            yaux=[yaux[m]+h*self.A[i][j]*P[l*len(yaux)+m] for m in range(len(yaux))]
                            
                            # aux=sp.lambdify(sp.Matrix(kasymlist),sp.Matrix(yaux))
                            sisrhs= rhs(t+h*self.c[i],*yaux)
                            #sisrhs=f(yaux) #parte derecha de la ecuación a resolver
                        for m in range(len(sisrhs)):
                            kas.append(sisrhs[m]-P[l*len(sisrhs)+m]) #se guarda de manera que las k's sean un 0 para resolver por newton
                        l=l+1
                    
                    return kas
                
                try:
                    aux=newton_krylov(sistarray,guess) # resolución numérica
                    #if (k+1)%1000==0:
                     #   print("{0:d} {1:.1f}s".format(k+1, time.time()-tiempo))
                except Exception:
                    try:
                        guess = zeros_like(kasymlist) 
                        aux=newton_krylov(sistarray,guess) # resolución numérica
                        #if (k+1)%1000==0:
                            #☻print("{0:d} {1:.1f}s".format(k+1, time.time()-tiempo))
                    except Exception as inst:    
                         print(inst)
                         print("Newton does not converge at time %f" %t) 
                         return rest,y 
                guess=aux.copy()

                #cálculo de y k+1 igual que en el caso anterior
                yaux=y[k]
                
                for i in range(len(self.b)):
                    yaux=[yaux[m]+h*self.b[i]*aux[i*len(yaux)+m] for m in range(len(yaux))]

                
                y.append(array(yaux))
                t+=h
                rest.append(t)
                
                
        return rest,y    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    