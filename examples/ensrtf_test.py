"""Ensemble filters in the 40-variable Lorenz model"""

import sys
from copy import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


Nenembles = 50               # number of ensembles
Npoints = 40                # number of spatial points
Nsteps = 1000               # number of steps in forward model
Nspinup = 100               # number of steps to spin up models

Nassim = 5                  # timesteps between assimilation cycles

obs_std  = 5.0              # standard error of observations
obs_var = obs_std**2
corr_length_scale = 5.0     # correlation filter length scale



class L40(object):
    
    def __init__(self,members=1,n=40,dt=0.05,F=8):
        self.n = n
        self.dt = dt
        self.dtx = dt
        self.x = np.random.normal(0.,0.1,size=(members,n))
        self.members = members
        self.F = F
    
    def dxdt(self):
        dxdt = np.zeros((self.members, self.n),'f8')
        for n in range(2,self.n-1):
            dxdt[:,n] = -self.x[:,n-2]*self.x[:,n-1] +  \
                        self.x[:,n-1]*self.x[:,n+1] - self.x[:,n] + self.F
        dxdt[:,0] = -self.x[:,self.n-2]*self.x[:,self.n-1] +  \
                self.x[:,self.n-1]*self.x[:,1] - self.x[:,0] + self.F
        dxdt[:,1] = -self.x[:,self.n-1]*self.x[:,0] + \
                self.x[:,0]*self.x[:,2] - self.x[:,1] + self.F
        dxdt[:,self.n-1] = -self.x[:,self.n-3]*self.x[:,self.n-2] + \
                            self.x[:,self.n-2]*self.x[:,0] - \
                            self.x[:,self.n-1] + self.F
        return dxdt
    
    def rk4step(self):
        h = self.dt; hh = 0.5*h; h6 = h/6.
        x = self.x
        dxdt1 = self.dxdt()
        self.x = x + hh*dxdt1
        dxdt2 = self.dxdt()
        self.x = x + hh*dxdt2
        dxdt = self.dxdt()
        self.x = x + h*dxdt
        dxdt2 = 2.0*(dxdt2 + dxdt)
        dxdt = self.dxdt()
        self.x = x + h6*(dxdt1 + dxdt + dxdt2)


def cov_filter(ndim, corr_length_scale):
    covlocal = np.zeros((ndim,ndim),'d')
    
    for j in range(ndim):
        for i in range(ndim):
            rr = float(i-j)
            if i-j < -(ndim/2): rr = float(ndim-j+i)
            if i-j > (ndim/2): rr = float(i-ndim-j)
            r = np.fabs(rr)/corr_length_scale
            if r < 1.0:
                covlocal[j,i] = (1.0-r)*np.cos(np.pi*r) + np.sin(np.pi*r)/np.pi
    
    return covlocal


# spinup truth and ensemble
truth = L40(n=Npoints)
ensemble = L40(n=Npoints, members=Nenembles)

# get covariance filter
covlocal = cov_filter(Npoints, corr_length_scale)

for nt in range(Nspinup): # spinup
    truth.rk4step()
    ensemble.rk4step()


# Set up figure and initialize lines on plot
fig = plt.figure(figsize=(10, 5))
ax = fig.add_axes([0.05, 0.05, 0.9, 0.90])

lines_en = ax.plot(np.zeros((Npoints, Nenembles), 'd'), '-k', lw=0.5, alpha=0.25)
lines_en_mean, = ax.plot(np.zeros((Npoints,), 'd'), '-k', lw=2)
lines_truth, = ax.plot(np.zeros((Npoints, ), 'd'), '-r', lw=2)
lines_obs, = ax.plot(np.zeros((Npoints, ), 'd'), 'go', ms=10)
ax.set_ylim(-10, 15)

null = np.ma.MaskedArray(np.zeros((Npoints,), 'd'), mask=True)


def anim_init():
    for ne in range(ensemble.members):
        lines_en[ne].set_ydata(null)
    lines_en_mean.set_ydata(null)
    lines_truth.set_ydata(null)
    lines_obs.set_ydata(null)
    
    output = copy(lines_en)
    output.append(lines_en_mean)
    output.append(lines_truth)
    output.append(lines_obs)
    return output


# boggie
def anim_frame(nt):
    
    # step truth and ensemble forward
    truth.rk4step()
    ensemble.rk4step()
    
    lines_obs.set_ydata(null)
    
    # Calculate mean and spread
    if nt % Nassim == 0:
        
        xm = ensemble.x.mean(axis=0)    # mean
        xp = ensemble.x - xm            # spread
        
        # Generate observations
        obs = truth.x[0] + obs_std * np.random.randn(Npoints)
        lines_obs.set_ydata(obs)
        
        # Assimilate data point-by-point
        for nobs in range(Npoints):
            if np.mod(nobs, 1) != 0 : continue
            # Estimate PH from spread at obs point
            PH = (xp.T*xp[:,nobs]).sum(axis=1) / (ensemble.members - 1.0)
            K = covlocal[nobs,:] * PH / (PH[nobs] + obs_var)  # Kalman gain
            # K = PH / (PH[nobs] + obs_var)  # Kalman gain
            alpha = ((PH[nobs] + obs_var) / PH[nobs]) \
                  * (1.0-np.sqrt(obs_var / (PH[nobs] + obs_var)))
            Kp = alpha * K      # the reduced gain for updating spread
            # print alpha
            
            # update mean and spread
            xm += K * (obs[nobs] - xm[nobs])
            for ne in range(ensemble.members):
                xp[ne, :] -= Kp * xp[ne, nobs]
        
        ensemble.x = xm + xp
    
    for ne in range(ensemble.members):
        lines_en[ne].set_ydata(ensemble.x[ne])
    lines_en_mean.set_ydata(ensemble.x.mean(axis=0))
    lines_truth.set_ydata(truth.x[0])
    
    output = copy(lines_en)
    output.append(lines_en_mean)
    output.append(lines_truth)
    output.append(lines_obs)
    return output
    

line_ani = animation.FuncAnimation(fig, anim_frame, interval=5, init_func=anim_init, blit=False)
plt.show()


    