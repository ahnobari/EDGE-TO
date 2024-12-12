import numpy as np
from ._base import Solver
from ..kernels._processors import StiffnessKernel
from scipy.sparse.linalg import splu, spsolve

class SPLU(Solver):
    def __init__(self, kernel: StiffnessKernel):
        super().__init__()
        
        if kernel.shape[0] > 3e6:
            raise ValueError("Currently we do not allow SuperLU for problem size bigger than 3M degrees of freedom. You can override this by passing a dummy kernel and overriding the kernel attribute.")
        
        self.kernel = kernel
    
    def solve(self, rhs, rho=None, **kwargs):
        
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
        
        LU = splu(K)
        
        out[self.kernel.non_con_map] = LU.solve(rhs_)
        
        residual = np.linalg.norm(rhs - self.kernel@out)/np.linalg.norm(rhs)
        
        return out, residual
    
    
class SPSOLVE(Solver):
    def __init__(self, kernel: StiffnessKernel):
        super().__init__()
        
        if kernel.shape[0] > 3e6:
            raise ValueError("Currently we do not allow spsolve for problem size bigger than 3M degrees of freedom. You can override this by passing a dummy kernel and overriding the kernel attribute.")
        
        self.kernel = kernel
    
    def solve(self, rhs, rho=None, **kwargs):
        
        if not rho is None:
            K = self.kernel.construct(rho)
        else:
            if not self.kernel.has_rho:
                raise ValueError("Solver requires a density vector to be passed or set on the kernel.")
            K = self.kernel.construct(self.kernel.rho)
        
        out = np.copy(rhs)
        
        if self.kernel.has_cons:
            K = K[:,self.kernel.non_con_map][self.kernel.non_con_map,:]
            rhs_ = rhs[self.kernel.non_con_map]
        
        
        out[self.kernel.non_con_map] = spsolve(K, rhs_)
        
        residual = np.linalg.norm(rhs - self.kernel@out)/np.linalg.norm(rhs)
        
        return out, residual