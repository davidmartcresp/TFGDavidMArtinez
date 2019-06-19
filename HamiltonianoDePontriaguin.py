# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 20:51:42 2019

@author: David
"""

from sympy import *
from rk import rk
import numpy as np
import math 
from sympy.utilities.lambdify import lambdify




# ecuaciones de movimiento del hamiltoniano H en la froma (\dot{Q},  \dot{P})
def poissonecs(H,X,P):
    """
    devuelve un sistema de ecuaciones para los estados X y los coestados P a partir del hamiltoniano H
    
    INPUT: H-- Symbol; Hamiltoniano
           X-- list de Symbol; Vector de estados
           P-- list de Symbol; Vector de coestados
    OUTPUT: res-- list de Symbol; representa la parte derecha del sistema [\dot{X},\dot{P}]=poissonecs(H,X,P)
    """
    res=[]
    for i in P:
        res.append(diff(H,i))
    for i in X:
        res.append(-diff(H,i))
    return res

#calcula el hamiltoniano de pontryagin para unas ecuaciones y coestados determinados
def hpontriagin(ecs,coestados,minim,multipl):
    """
    devuelve un el hamiltoniano de Pontriaguin asociado al sistema 
    
    INPUT: ecs-- list de Symbol; rhs de las ecuaciones
           coestados-- list de Symbol; Vector de coestados
           minim --  Symbol; integrando del coste a minimizar
           multipl -- Symbol; El mutliplicador de minim, \Pi_0
           
    OUTPUT: H-- Symbol; el hamiltoniano de Pontriaguin asociado al sistema
    """
    H=0
    for i in range(len(ecs)):
        H+=coestados[i]*ecs[i]
    H+=multipl*minim
    return H

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    EJEMPLO DE SISTEMA HÍBRIDO
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# definición de variables
t=Symbol('t')
ep=Symbol('\epsilon')
P1,Q1,P2,Q2,p1,q1,p2,q2=symbols('P_1 Q_1 P_2 Q_2 p_1 q_1 p_2 q_2')
Bx,By=symbols('B_x By')
Pi0,Pi1,Pi2,Pi3,Pi4,Pi5,Pi6,Pi7,Pi8=symbols('\Pi_0 \Pi_1 \Pi_2 \Pi_3 \Pi_4 \Pi_5 \Pi_6 \Pi_7 \Pi_8')


"""
        Hamiltoniano del sistema híbrido
"""
    
#funciones de acoplo
f=epsilon*Q1
g=epsilon*(P1**2+P2**2-2*Q1*Q2)/2

#hamiltoniano cuantico
sx=p1*p2+q1*q2
sy=p2*q1-p1*q2
sz=(q1**2+p1**2-q2**2-p2**2)


#hamiltonanao del sistema
H=(Q1*Q1+Q2*Q2+P1*P1+P2*P2)/2+Bx*f*sx-By*g*sy


#variables y variables conjugadas
X=[Q1,Q2,q1,q2]
P=[P1,P2,p1,p2]




""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                          PRINCIPIO DEL MÁXIMO
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#funcional de coste
L=(Bx**2+By**2)/2

#hamiltoniano de pontrigin 
co=[Pi1,Pi2,Pi3,Pi4,Pi5,Pi6,Pi7,Pi8]
Hp=(hpontriagin(poissonecs(H,X,P),co,L,Pi0))


#controles óptimos
Bxoptimo=solve(diff(Hp,Bx),Bx)
Byoptimo=solve(diff(Hp,By),By)
"""
WARNING: solo se comprueba la condición de extremal.
"""

#sistema con controles óptimos
sist=poissonecs(Hp.subs(Bx,Bxoptimo[0]).subs(By,Byoptimo[0]).subs(Pi0,-1),X+P,co)  