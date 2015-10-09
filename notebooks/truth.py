import numpy as np

# Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut
# labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco
# laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in
# voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat
# non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
# Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut
# labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco
# laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in
# voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat
# non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
# Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut
# labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco
# laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in
# voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat
# non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
# Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut
# labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco
# laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in
# voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat
# non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
# Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut
# labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco
# laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in
# voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat
# non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
# Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut
# labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco
# laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in
# voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat
# non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
# Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut
# labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco
# laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in
# voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat
# non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
# Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut
# labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco
# laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in
# voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat
# non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
# Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut
# labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco
# laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in
# voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat
# non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.



class TruthA(object):
    
    A  = np.array([ 3.88092463,  5.27598177,  3.94063443,  -2.40558618,  3.46678858])
    Rd = np.array([ 0.35888211,  0.01272478,  0.21547899,  0.06001511,  0.99942273])/3.0
    xo = np.array([ 0.56722137,  0.75425931,  0.34285972,  0.37095792,  0.80756739])
    yo = np.array([ 0.41139973,  0.20957578,  0.80260264,  0.43739268,  0.48978928])

    def _z(self, x, y):
        z = np.sin((x+y)*10.0)
        for xo, yo, Rd, A in zip(self.xo, self.yo, self.Rd, self.A):
            z += A * np.exp(-((x-xo)**2 + (y-yo)**2)/Rd)
        return z

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.z = self._z(x, y)
        
    def sample(self, N):
        x = np.random.rand(N)
        y = np.random.rand(N)
        z = self._z(x, y)
        return x, y, z


class TruthB(object):
    
    A  = np.array([-3.88092463,  2.27598177,  3.94063443,  -2.40558618,  3.46678858])
    Rd = np.array([ 0.65888211,  0.01272478,  0.61547899,  0.60001511,  0.99942273])
    xo = np.array([ 0.56722137,  0.75425931,  0.34285972,  0.37095792,  0.80756739])
    yo = np.array([ 0.41139973,  0.20957578,  0.80260264,  0.43739268,  0.48978928])

    def _z(self, x, y):
        z = np.sin(x*100.0)
        for xo, yo, Rd, A in zip(self.xo, self.yo, self.Rd, self.A):
            z += A * np.exp(-((x-xo)**2 + (y-yo)**2)/Rd)
        return z

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.z = self._z(x, y)
        
    def sample(self, N):
        x = np.random.rand(N)
        y = np.random.rand(N)
        z = self._z(x, y)
        return x, y, z


