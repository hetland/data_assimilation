"""Ensemble filters in the 40-variable Lorenz model"""
import numpy as np
import sys

import matplotlib
matplotlib.use('TkAgg') # do this before importing pylab
import matplotlib.pyplot as plt

nens = 30               # number of ensembles

oberrstdev = 3.0        # std of observations


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



# spinup for true run.
ntstart = 100
model = L40()
for nt in range(ntstart): # spinup
   model.rk4step()


xx = []
tt = []
ntimes = 1000
for nt in range(ntimes):
    model.rk4step()
    xx.append(model.x[0]) # single member
    tt.append(float(nt)*model.dt)

xtruth = np.array(xx, 'd')
timetruth = np.array(tt, 'd')
xtruth_mean = xtruth.mean()
xprime = xtruth - xtruth_mean
xvar = np.sum(xprime**2,axis=0)/(ntimes-1)
xtruth_stdev = np.sqrt(xvar.mean())


dtassim = 1.0*model.dt
ensemble = L40(members=nens)
oberrvar = oberrstdev**2
ndim = ensemble.n

obs = xtruth + oberrstdev*np.random.standard_normal(size=xtruth.shape)

# spinup fcst model
tstart = 100
ntstart = int(tstart/model.dt)
ntot = xtruth.shape[0]
nspinup = ntot/2
print 'ntstart, nspinup, ntot =',ntstart,nspinup,ntot
for n in range(ntstart):
   ensemble.rk4step() # perfect model

nsteps = dtassim/model.dt
if nsteps % 1  != 0:
    raise ValueError, 'assimilation interval must be an integer number of model time steps'
else:
   nsteps = int(nsteps)


def serial_ensrf(ensemble,xmean,xprime,obs,oberrvar):
    """serial potter method"""
    # random order
    #indx = np.argsort(uniform(size=ensemble.n))
    #for nob in indx:
    for nob in range(ensemble.n):
        pfht = np.sum(np.transpose(xprime)*xprime[:,nob],1)/float(ensemble.members-1)
        # print pfht.shape
        kfgain = covlocal[nob,:]*pfht/(pfht[nob]+oberrvar)
        gainfact = ((pfht[nob]+oberrvar)/pfht[nob])*\
                   (1.-np.sqrt(oberrvar/(pfht[nob]+oberrvar)))
        reducedgain = gainfact*kfgain
        # update mean
        xmean = xmean + kfgain*(obs[nob]-xmean[nob])
        # update perturbations
        for nm in range(ensemble.members):
            xprime[nm] = xprime[nm] - reducedgain*xprime[nm,nob]
    return xmean, xprime

def serial_enkf(ensemble,xmean,xprime,obs,oberrvar):
    """serial perturbed obs EnKF"""
    obperts = np.sqrt(oberrvar)*np.random.standard_normal(size=xprime.shape)
    obpertmean = obperts.mean(axis=0)
    obperts = obperts - obpertmean
    for nob in range(ensemble.n):
        pfht = np.sum(np.transpose(xprime)*xprime[:,nob],1)/float(ensemble.members-1)
        kfgain = pfht/(pfht[nob]+oberrvar)
        # update mean
        xmean = xmean + kfgain*(obs[nob]-xmean[nob])
        # update perturbations
        for nm in range(ensemble.members):
            xprime[nm] = xprime[nm] + kfgain*(obperts[nm,nob]-xprime[nm,nob])
    return xmean, xprime



# run assimilation.
xsave = ensemble.x.copy()

covinflate = 1.0
corrl = 1.0


ensemble.x = xsave
covlocal = np.zeros((ndim,ndim),'d')

for j in range(ndim):
    for i in range(ndim):
        rr = float(i-j)
        if i-j < -(ndim/2): rr = float(ndim-j+i)
        if i-j > (ndim/2): rr = float(i-ndim-j)
        r = np.fabs(rr)/corrl
        if r < 1.0:
            covlocal[j,i] = (1.0-r)*np.cos(np.pi*r) + np.sin(np.pi*r)/np.pi

fcsterr = []
fcstsprd = []
analerr = []
analsprd = []

fig = plt.figure(figsize=(10, 5))
ax = fig.add_axes([0.05, 0.05, 0.9, 0.90])

lines_en = plt.plot(np.zeros((40, nens), 'd'), '-k', alpha=0.5)
lines_truth, = plt.plot(np.zeros((40, ), 'd'), '-r', lw=2)
ax.set_ylim(-5, 15)

def run_ensemble(*args):
    for nassim in range(0,ntot,nsteps):
        # assimilate obs
        xmean = ensemble.x.mean(axis=0)
        xprime = ensemble.x - xmean
        xprime = covinflate*xprime
        ferr = ((xmean - xtruth[nassim])**2).mean()
        fsprd = (xprime**2).mean()
        if nassim >= nspinup:
            fcsterr.append(ferr); fcstsprd.append(fsprd)
        xmean, xprime = serial_ensrf(ensemble,xmean,xprime,obs[nassim,:],oberrvar)
        # run forecast model.
        aerr = ((xmean - xtruth[nassim])**2).mean()
        asprd = (xprime**2).mean()
        if nassim >= nspinup:
            analerr.append(aerr); analsprd.append(asprd)
        print nassim,timetruth[nassim],ferr,fsprd,aerr,asprd
        ensemble.x = xmean + xprime
        for n in range(nsteps):
            ensemble.rk4step() # perfect model
        
        for n, xi in enumerate(ensemble.x):
            lines_en[n].set_ydata(xi)
        lines_truth.set_ydata(xtruth[nassim])
        
        fig.canvas.draw()

fig.canvas.manager.window.after(100, run_ensemble)
plt.show()
    


fcsterr = np.array(fcsterr)
fcstsprd = np.array(fcstsprd)
analerr = np.array(analerr)
analsprd = np.array(analsprd)
fstdev = np.sqrt(fcstsprd.mean())
astdev = np.sqrt(analsprd.mean())
print corrl,covinflate,np.sqrt(analerr.mean()),astdev