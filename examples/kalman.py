# kalman.py
#
# Toy ocean model with the kalman filter.
#
# (c) Robert Hetland, 2006-10-31

from __future__ import print_function

from numpy import *

import matplotlib.pyplot as pl
import matplotlib.animation as animation

#####################
# Data and data stats

wavenum = 50.0                      # Wavenumber of 'truth'

Ce = 0.5                            # Data error varience (diagonal Ce)
Ro = pi/wavenum                     # Radius of influence
c_data = 1.0                        # Data phase speed
                                    
M = 50                              # Number of data points
N = 10                              # Number of data times
                                    
T = 5.0                             # time of integration
Tassim = 0.05                       # assimilation time preiod
                                    
dx = 0.005                          # Grid resolution
dt = dx/1.

c  = 1.0                            # Phase speed
kappa = 0.0001                       # diffusion (for numerical stability)

Qfo = 0.5                           # Forcing error variance
Qb = 50.0                             # Boundary condition error variance

######################

def oi1d(xmidx,dm,x,bvar,Eo):
    # function A, Ea = oi1d(xm,dm,x,Ro,Bvar,Dvar)
    #   Returns analysis and error field from gridded data.
    #
    # Inputs:
    #   xmidx:   X position of data points (indices)
    #   dm:      Value of data at points xm.
    #   x:       Grid on which to interpolate
    #   Bvar:    Background error variance
    #   Eo       Data error
    # Outputs:
    #   A:       Analysis field
    #   Ea:      Analysis error variance
    
    Bi = bvar[xmidx, :]
    B = bvar[xmidx, :][:, xmidx]
    
    O = Eo**2 * eye(len(xmidx));    # Obs error is diagonal (uncorrelated errors)
    W = dot(linalg.inv(B + O), Bi) # Optimal weights (Kalman gain)
    A = dot(dm, W)             # Analysis field
    Ea = ( bvar - dot(W.T, Bi) )  # Analysis error variance
    return A.flatten(), Ea


fig = pl.figure(figsize=(20, 10))
ax = fig.add_axes([0.03, 0.05, 0.6, 0.90])
axp = fig.add_axes([0.68, 0.15, 0.3, 0.65])
ax.set_ylim(-3, 3)
title = ax.set_title('')
ax.grid(True)


idx = 0
tdata = 0.0
# Define the grid
x = mgrid[0.0:1.0+dx:dx]                        # location of grid points
t = arange(dt,T+dt,dt)

# time steps between assimilation cycles
assimn = int(Tassim/dt)

# Set up measurement locations
# xidx = arange(50, 75)
# xidx = arange(50, len(x)-50)
xidx = arange(len(x))
random.shuffle(xidx)
xmidx = xidx[:M]
# xmidx = arange(60, 60+M)
xm = x[xmidx]                                  # Data positions

# Set up the Model forcing error covariance 
# (Also initial background error covariance)
Qf = Qfo*exp(-abs(x[:,newaxis]- x[newaxis,:])**2 / Ro**2)

# Initial condition (Initial analysis field)
dm = sin(xm*wavenum) + sqrt(Ce)*random.randn(M)  # initial data at t=0
w_ini, Ci = oi1d(xmidx, dm, x, Qf, Ce)          # initial field and error covariance

# Parameters for forward model
fac = c*dt/dx
fak = kappa*dt/dx**2
# Initialize w
w = w_ini.copy()
# Initialize P
P  = Ci.copy()


mu = 0.0
phi = 0.5


line_w, = ax.plot(x, w, '-k', lw=2)
line_true, = ax.plot(x, sin((x)*wavenum), '-r')
line_data, = ax.plot(zeros((M), 'd'), zeros((M), 'd'), 'g.', markersize=10.0)
line_errp, = ax.plot(x, zeros_like(x), '-', color='0.5')
line_errm, = ax.plot(x, zeros_like(x), '-', color='0.5')
pc = axp.imshow(sqrt(abs(P)), vmin=0, vmax=1.0, cmap='Blues')

def init_line():
    line_w.set_ydata(ma.MaskedArray(w, mask=True))
    line_true.set_ydata(ma.MaskedArray(x, mask=True))
    
    line_errp.set_ydata(ma.MaskedArray(x, mask=True))
    line_errm.set_ydata(ma.MaskedArray(x, mask=True))
    
    line_data.set_xdata(ma.MaskedArray(x, mask=True))
    line_data.set_ydata(ma.MaskedArray(x, mask=True))
    
    pc.set_data(ma.MaskedArray(P, mask=True))
    
    return (line_w, line_true, line_data, line_errp, line_errm, pc)

# Boogie -- run the forward dynamic and error variance models
def update_line(n):
    global w, mu, P, tdata, dm
    # Forward integrate dynamic model
    w[1:] = w[1:] - fac*(diff(w))
    # add in diffusion
    w[1:-1] = w[1:-1] + fak*(w[:-2]-2*w[1:-1]+w[2:])
    # add in forcing noise
    w[1:] = w[1:] + dt*sqrt(Qfo) #*random.randn(*w[1:].shape)   # *1.414*sin(t[n]*wavenum/5.0)
    # Set boundary condition, with noise
    
    eps = random.randn(1)
    w[0] = sin(-t[n]*wavenum) + sqrt(Qb)*(phi*mu + eps) / sqrt(1.0 - phi)
    mu = eps
    
    w = w + zeros_like(w)  # only done to ensure plotting (bug in mpl?)
    
    # Forward integrate error covariance
    P[1:,1:] = P[1:,1:] \
           - fac*0.5*(diff(P[:,1:], axis=0) + diff(P[:,:-1], axis=0)) \
           - fac*0.5*(diff(P[1:,:], axis=1) + diff(P[:-1,:], axis=1)) \
           + dt*0.5*(Qf[1:,1:] + Qf[:-1,:-1])
    P[1:-1,1:-1] = P[1:-1,1:-1] \
           + fak*(P[:-2, 1:-1]-2*P[1:-1, 1:-1]+P[2:, 1:-1]) \
           + fak*(P[1:-1, :-2]-2*P[1:-1, 1:-1]+P[1:-1, 2:])
    P[0,0] = c*Qb


    # Assimilation cycle -- optimally interpolate data
    if not mod(n, assimn):
        dm = sin((xm - c_data * t[n])*wavenum) + sqrt(Ce)*random.randn(M)
        ai = dm - w[xmidx]
        wa, Pa = oi1d(xmidx, ai, x, P, Ce)
        w = w+wa
        P = Pa
        print('Assimilation cycle at %4.2f, mean error %f' % (t[n], P.diagonal().sum()))
        
        tdata = t[n]
    
    # Plot next frame in analysis
    err = sqrt(diag(P))
    tru = sin((x - t[n])*wavenum)
    
    line_w.set_ydata(w)
    line_true.set_ydata(tru)
    
    line_errp.set_ydata(w+err)
    line_errm.set_ydata(w-err)
    
    line_data.set_xdata(x[xmidx] + c*(t[n]-tdata))
    line_data.set_ydata(dm)
    
    pc.set_data(sqrt(abs(P)))
    
    # title.set_text('%f' % t[n])
    
    return (line_w, line_true, line_data, line_errp, line_errm, pc)
    

line_ani = animation.FuncAnimation(fig, update_line, interval=5, init_func=init_line, blit=False)
#line_ani.save('lines.mp4')

pl.show()

