import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
from IPython.display import HTML
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from matplotlib import cm, colors
import matplotlib as mpl
import matplotlib.colors as mcolors
from sympy import *
from rk import rk
import math 
from sympy.utilities.lambdify import lambdify




"""
        Ejecuta este código junto a rk.py y se mostrarán varios ejemplos de problemas de control para el
        sistema de ejemplo en el Trabajo de Fin De grado en matemáticas de David Martínez Crespo 
        Universidad de Zaragoza:
        
            Formalismo geométrico de la Mecánica Cuántica y sus aplicaciones a modelos moleculares 

"""

class bloch:
    def __init__(self,ax,fig):
        """ 
        mpl.rcParams['legend.fontsize'] = 10
          
        fig = plt.figure(figsize=(9, 9))

        plt.style.use('seaborn-darkgrid')

        
        
        ax = fig.add_subplot(111, projection='3d')
        """
        self.ax=ax

        """
        # Make data
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x1 = 1 * np.outer(np.cos(u), np.sin(v))
        y1 = 1 * np.outer(np.sin(u), np.sin(v))
        z1 = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_wireframe(x1, y1, z1, color='grey', rstride=4, cstride=4,
                          alpha=0.25, label = 'Esfera de Bloch')

        ax.legend()
        
        plt.title("Trayectoria cuántica", fontsize=16)
        """
        self.fig=fig
        
    def grid(self,encender=True):
        ax=self.ax
        ax.grid(encender)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        return self.fig
        
    
    def projection(self,q1,p1,q2,p2):
        norm=q1**2+q2**2+p2**2+p1**2
        x=2*(q1*q2+p1*p2)
        y=2*(q1*p2-q2*p1)
        z=(q1**2+p1**2-q2**2-p2**2)
        return x,y,z
        
    def rotate(self,elevation=0,azimutal=0 ):
        ax=self.ax
        ax.view_init(azim=ax.azim+azimutal,elev=ax.elev+elevation)
        return self.fig
    
    def traj4c(self,r,label='Trayectoria'):
        ax=self.ax
        x=[]
        y=[]
        z=[]
        for k in r:
            xa,ya,za=self.projection(*k)
            x.append(xa)
            y.append(ya)
            z.append(za)
            
        ax.plot(x, y, z, label=label)
        #ax.legend()
        
        return self.fig
    
    
    def traj4cdeg(self,r,cmap,label='Trayectoria'):
        ax=self.ax
        x=[]
        y=[]
        z=[]
        for k in r:
            xa,ya,za=self.projection(*k)
            x.append(xa)
            y.append(ya)
            z.append(za)
            
        for i in range(len(x)-1):
            ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=cmap(i/len(x)))
        #ax.legend()
        
        return self.fig
    
    def show(self):
        plt.show()
    
    def scatter(self, r,color="auto",label="t1"):
        ax=self.ax
        xa,ya,za=self.projection(*r)
        if color=="auto":
            ax.scatter(xa,ya,za,label=label)
        else:
            ax.scatter(xa,ya,za,label=label,c=color,s=50)
        #ax.legend()
        
        return self.fig
                

## código de los plots
def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0,Tfinal),
        linewidth=3, alpha=1.0,ax =None):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    if ax is None:
        ax=plt.gca()
    
    line=ax.add_collection(lc)
    #fig.colorbar(line, ax=ax)
    
    
    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap




def uneColormaps(cmap1,cmap2,fraccion=500,num=1000):
    """Return a sum of colomaps"""
    cmap1 = plt.cm.get_cmap(cmap1)
    colors1 = cmap1(np.linspace(0., 1, fraccion))
    cmap2 = plt.cm.get_cmap(cmap2)
    colors2 = cmap2(np.linspace(0., 1, num-fraccion))
    colors=[i for i in colors1]+[j for j in colors2 ]
        
    return mcolors.LinearSegmentedColormap.from_list(cmap1.name + cmap2.name, colors, len(colors))

def plots(estados,tf,Q0P0,QfPf,QfPfquantum):
    #figuras
    plt.style.use('default')
    cmap1=truncate_colormap(plt.get_cmap('jet'),0,1,1000)
    nuevoestados=estados#+estados2
    
    fig = plt.figure(figsize=(14, 6))
    
    ax1 = fig.add_subplot(1, 2, 2,projection="3d")
    
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x1 = 1 * np.outer(np.cos(u), np.sin(v))
    y1 = 1 * np.outer(np.sin(u), np.sin(v))
    z1 = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax1.plot_wireframe(x1, y1, z1, color='grey', rstride=4, cstride=4,
                      alpha=0.25, label = 'Esfera de Bloch')
    esfera=bloch(ax1,fig)
        # Hide grid lines
    ax1.grid(False)
    
    # Hide axes ticks
    ax1.set_xticks([-1,0,1])
    ax1.set_yticks([-1,0,1])
    ax1.set_zticks([-1,0,1])
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    
    ax1.text3D(0,0,-1.5,r'$\left|0\right\rangle$', size=20)
    ax1.text3D(0,0,1.2,r'$\left|1\right\rangle$', size=20)
    
    trajcuan=[[r[2],r[6],r[3],r[7]] for r in nuevoestados]
    #trajcuan2=[[r[2],r[6],r[3],r[7]] for r in estados2]
        
    esfera.traj4cdeg(trajcuan,cmap=cmap1)
    #esfera.traj4c(trajcuan2,"tf="+str(2*Tfinal))
    r=Q0P0
    esfera.scatter([r[2],r[6],r[3],r[7]],"red","t0")
    r=QfPfquantum
    esfera.scatter([r[0],r[1],r[2],r[3]],"blue","t1")
    
    esfera.scatter([0,0,0,1],"green","t1")
    
    
    

    ax2 = fig.add_subplot(2, 2, 1)
    ax3 = fig.add_subplot(2, 2, 3)
    
    
    
    #primer tramo
    x = np.array( [z[0] for z in nuevoestados])
    y = np.array([z[1] for z in nuevoestados])
    path = mpath.Path(np.column_stack([x, y]))
    verts = path.interpolated(steps=3).vertices
    x, y = verts[:, 0], verts[:, 1]
    z = np.linspace(0, Tfinal, len(x))
    
    colorline(x, y, z, cmap=cmap1, linewidth=3,ax=ax2)
    
    
    
    
    ax2.set_xlim(-0.1+x.min(), x.max()+0.1)
    ax2.set_ylim(y.min()-0.2,y.max()+0.2)
    
    xa=ax2.get_xaxis()
    xa.set_ticks_position("top")
    ax2.set_xlabel("$Q_1$")
    ax2.set_ylabel("$Q_2$")
    
    
    
    x = np.array( [z[4] for z in nuevoestados])
    y = np.array([z[5] for z in nuevoestados])
    path = mpath.Path(np.column_stack([x, y]))
    verts = path.interpolated(steps=3).vertices
    x, y = verts[:, 0], verts[:, 1]
    z = np.linspace(0, Tfinal, len(x))
    p1=ax2.scatter(Q0P0[0],Q0P0[1],c='red')
    p2=ax2.scatter(QfPf[0],QfPf[1],c='blue')

    
    fig.legend((p1, p2,p3), ('Punto inicial','Punto intermedio' ,'Punto final'), 'upper right')
    
    lc=colorline(x, y, z, cmap=cmap1, linewidth=3,ax=ax3)
    ax3.set_xlim(-0.1+x.min(), x.max()+0.1)
    ax3.set_ylim(y.min()-0.2,y.max()+0.2)
    
    ax3.set_xlabel("$P_1$")
    ax3.set_ylabel("$P_2$")
    ax3.scatter(Q0P0[4],Q0P0[5],c='red')
    ax3.scatter(QfPf[2],QfPf[3],c='blue')
    #fig.suptitle("Trayectorias en cada proyección", fontsize=16)








    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.97, 0.15, 0.01, 0.7])
    
    norm = mpl.colors.Normalize(vmin=0, vmax=Tfinal)
    cb1 = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap1,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('tiempo')
    return fig



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

"""##############################################################
                    EJEMPLO DE SISTEMA HÍBRIDO
###############################################################"""

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
epsilon=1
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




""" ##################
                          PRINCIPIO DEL MÁXIMO
##################################################################"""
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

sistevaluable=lambdify([t]+X+P+co,Matrix(sist))

A=np.array([[0.5]])
b=np.array([1])
c=np.array([0.5])
# El tiempo t_f viene indicado por
rkGL=rk(A,b,c)

Tfinal=4

# La condición inicial está guardada en Q0P0 con el formato
# [Q_1, Q_2, q_1, q_2, P_1, P_2, p_1, p_2]

Q0P0=[1, 0, 1/2, 1/2, -0.5, -1, 1/2, 1/2]

# La condición final está guardada en QfPf con el formato
# [Q_1, P_1, Q_2, P_2], la condición cuántica se especifica en un comentario

QfPf=[1, 1, 1 , 0]# \times |1 >

# La condición inicial del sistema extendido de Pontriguin se guarda en u0
# coestados es una variable auxiliar que guarda la condición inicial de los coestados 
# en formato [\Pi_1,\Pi_2,\Pi_3,\Pi_4,\Pi_5,\Pi_6,\Pi_7,\Pi_8]

coestados=np.array([1.23989, -4.04735, 4.15565, 4.95752, -1.98568, 0.228084, 6.60882, 2.50435])
u0=np.array([m for m in Q0P0]+[l for l in coestados])



tes,estados=rkGL(sistevaluable,u0,tf=Tfinal,N=500,t0=0)
fig=plots(estados,Tfinal,Q0P0,QfPf,[1,0,0,0])




Tfinal=5
Q0P0=[1, 1, 1, 0, 1, 0, 0, 0]
QfPf=[0.5, 0.5, 0.5, 1]# \times |0>
coestados=np.array([ -0.643419, 0.526499, 8.38667, -0.057135, -2.68758, 0.385212, 0, 1.08133])
u0=np.array([m for m in Q0P0]+[l for l in coestados])


tes,estados=rkGL(sistevaluable,u0,tf=Tfinal,N=500,t0=0)
fig=plots(estados,Tfinal,Q0P0,QfPf,[0,0,0,1])



Tfinal=5
Q0P0=[1, 1, 1, 0, 1, 0, 0, 0]
QfPf=[0.5, 0.5, 0.75, 1]# \times |0>
coestados=np.array([-0.0625046, 0.0840827, 6.88043, -0.209125, -2.6056, 0.255122, 0, 0.959606])
u0=np.array([m for m in Q0P0]+[l for l in coestados])


tes,estados=rkGL(sistevaluable,u0,tf=Tfinal,N=500,t0=0)
fig=plots(estados,Tfinal,Q0P0,QfPf,[0,0,0,1])



Tfinal=5
Q0P0=[1, 1, 1, 0, 1, 0, 0, 0]
QfPf=[0.5, 0.5, 0.5, 0.5]# \times |0>
coestados=np.array([1.19383, -1.06154, 10.7773, -0.46899, -2.95481, -0.381508, 0, -0.314177])
u0=np.array([m for m in Q0P0]+[l for l in coestados])


tes,estados=rkGL(sistevaluable,u0,tf=Tfinal,N=500,t0=0)
fig=plots(estados,Tfinal,Q0P0,QfPf,[0,0,0,1])



Tfinal=5
Q0P0=[0.5, 0.5, 0,0, 0.5, 1, 1, 0]
QfPf=[-0.25, -1, 0.5, 0.5] #\times |1>  
coestados=np.array([2.12191, -2.00047, 0, 0.723979, 3.5473, 0.466018, 3.55722, 0.913273])
u0=np.array([m for m in Q0P0]+[l for l in coestados])


tes,estados=rkGL(sistevaluable,u0,tf=Tfinal,N=500,t0=0)
fig=plots(estados,Tfinal,Q0P0,QfPf,[1,0,0,0])

































        
     
        
    



