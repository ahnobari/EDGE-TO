from sksparse.cholmod import cholesky
import numpy as np
from ._base import Solver
from ..kernels._processors import StiffnessKernel

class CHOLMOD(Solver):
    def __init__(self, kernel: StiffnessKernel):
        super().__init__()
        
        if kernel.shape[0] > 3e6:
            raise ValueError("Currently we do not allow CHOLMOD for problem size bigger than 3M degrees of freedom. You can override this by passing a dummy kernel and overriding the kernel attribute.")
        
        self.kernel = kernel
        self.factor = None
        self.factorized = False
        self.n_desvars = len(kernel.elements)
        
    def reset(self):
        self.factor = None
        self.factorized = False
        
    def initialize(self):
        
        rho_temp = np.ones(self.n_desvars, dtype=self.kernel.nodes.dtype)
        K = self.kernel.construct(rho_temp)
        K = self.kernel.construct(rho_temp) # Twice to get correct pointers
        if self.kernel.has_cons:
            K = K[:,self.kernel.non_con_map][self.kernel.non_con_map,:]
            
        self.factor = cholesky(K)
        self.factorized = True
    
    def solve(self, rhs, rho=None, **kwargs):
        
        if not self.factorized:
            self.initialize()
        if not rho is None:
            K = self.kernel.construct(rho)
        else:
            if not self.kernel.has_rho:
                raise ValueError("Solver requires a density vector to be passed or set on the kernel.")
            K = self.kernel.construct(self.kernel.rho)
        
        out = np.zeros_like(rhs)
        
        if self.kernel.has_cons:
            K = K[:,self.kernel.non_con_map][self.kernel.non_con_map,:]
            rhs_ = rhs[self.kernel.non_con_map]
        
        self.factor.cholesky_inplace(K)
        
        out[self.kernel.non_con_map] = self.factor(rhs_)
        
        residual = np.linalg.norm(rhs - self.kernel@out)/np.linalg.norm(rhs)
        
        return out, residual