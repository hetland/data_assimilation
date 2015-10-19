
import numpy as np
from scipy.integrate import odeint

def lorenzatt(X, rho=28.0, sigma=10.0, beta=3.0/8.0):
    '''
    Lorenz Attractor equations (to be solved with scipy.integrate.odeint)
    
    x' = sigma*(y-x)
    y' = x*(rho - z) - y
    z' = x*y - beta*z
    '''
    
    dx = np.zeros((3), 'd')
    dx[0] = sigma*(X[1] - X[0]);
    dx[1] = X[0]*(rho - X[2]) - X[1]
    dx[2] = X[0]*X[1] - beta*X[2]
    return dx


class L40(object):
    '''Lorenz 40 model of zonal atmospheric flow'''
    
    def __init__(self, members=1, n=40, dt=0.05, F=8):
        self.n = n
        self.dt = dt
        self.dtx = dt
        self.x = np.random.normal(0., 0.1, size=(members, n))
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






if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    t = np.linspace(0, 100.0, 1e5)
    X0 = [0, 1, 1.05]
    x = odeint(lorenzatt, X0, t);
    print x.shape
    plt.plot(t, x[:,0], '-k')
    
    X0 = [0, 1., 1.08]
    x = odeint(lorenzatt, X0, t);
    print x.shape
    plt.plot(t, x[:,0], '-r')
    
    plt.show()
    