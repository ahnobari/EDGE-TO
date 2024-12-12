import cupy as cp
from ._base import Solver
from ..kernels._cuda import StiffnessKernel as CuStiffnessKernel
from ..kernels._cuda import GeneralStiffnessKernel as CuGeneralKernel
from ..kernels._cuda import UniformStiffnessKernel as CuUniformKernel
from cupyx.scipy.sparse.linalg import cg, gmres, spsolve
import time
import logging
loger = logging.getLogger(__name__)

class CG(Solver):
    def __init__(self, kernel: CuStiffnessKernel, maxiter=1000, tol=1e-5, matrix_free=None):
        super().__init__()
        self.kernel = kernel
        self.last_x0 = None
        self.tol = tol
        self.maxiter = maxiter
        
        if isinstance(kernel, CuGeneralKernel) or isinstance(kernel, CuUniformKernel):
            if matrix_free is None:
                matrix_free = False
                loger.warning("Matrix free is set to False for general and uniform kernels. This is recommended for speed. Will use more memory.")
            elif matrix_free:
                loger.warning("Matrix free is set to True for general and uniform kernels. This is not recommended for speed. Will use less memory however.")
        elif matrix_free is None:
            matrix_free = True
        
        self.matrix_free = matrix_free
        
        if not self.matrix_free:
            self.K = None
    
    def solve(self, rhs, rho=None, use_last=True):
        
        if self.kernel.has_rho or (not rho is None):
            if rho is not None:
                self.kernel.set_rho(rho)
            if use_last:
                if self.matrix_free:
                    out = cg(self.kernel, rhs, x0=self.last_x0, tol = self.tol, maxiter=self.maxiter)[0]
                    residual = cp.linalg.norm(rhs - self.kernel@out)/cp.linalg.norm(rhs)
                else:
                    self.K = self.kernel.construct(self.kernel.rho)
                    out = cg(self.K, rhs, x0=self.last_x0, tol = self.tol, maxiter=self.maxiter)[0]
                    residual = cp.linalg.norm(rhs - self.K@out)/cp.linalg.norm(rhs)
                self.last_x0 = out
            else:
                if self.matrix_free:
                    out = cg(self.kernel, rhs, tol = self.tol, maxiter=self.maxiter)[0]
                    residual = cp.linalg.norm(rhs - self.kernel@out)/cp.linalg.norm(rhs)
                else:
                    self.K = self.kernel.construct(self.kernel.rho)
                    out = cg(self.K, rhs, tol = self.tol, maxiter=self.maxiter)[0]
                    residual = cp.linalg.norm(rhs - self.K@out)/cp.linalg.norm(rhs)
        else:
            raise ValueError("Solver requires a density vector to be passed or set on the kernel.")
        
        return out, residual

class GMRES(Solver):
    def __init__(self, kernel: CuStiffnessKernel, maxiter=1000, tol=1e-5, matrix_free=None):
        super().__init__()
        self.kernel = kernel
        self.last_x0 = None
        self.tol = tol
        self.maxiter = maxiter
        
        if isinstance(kernel, CuGeneralKernel) or isinstance(kernel, CuUniformKernel):
            if matrix_free is None:
                matrix_free = False
                loger.warning("Matrix free is set to False for general and uniform kernels. This is recommended for speed. Will use more memory.")
            elif matrix_free:
                loger.warning("Matrix free is set to True for general and uniform kernels. This is not recommended for speed. Will use less memory however.")
        elif matrix_free is None:
            matrix_free = True
        
        self.matrix_free = matrix_free
        
        if not self.matrix_free:
            self.K = None
    
    def solve(self, rhs, rho=None, use_last=True):
        
        if self.kernel.has_rho or (not rho is None):
            if rho is not None:
                self.kernel.set_rho(rho)
            if use_last:
                if self.matrix_free:
                    out = gmres(self.kernel, rhs, x0=self.last_x0, tol = self.tol, maxiter=self.maxiter)[0]
                    residual = cp.linalg.norm(rhs - self.kernel@out)/cp.linalg.norm(rhs)
                else:
                    self.K = self.kernel.construct(self.kernel.rho)
                    out = gmres(self.K, rhs, x0=self.last_x0, tol = self.tol, maxiter=self.maxiter)[0]
                    residual = cp.linalg.norm(rhs - self.K@out)/cp.linalg.norm(rhs)
                self.last_x0 = out
            else:
                if self.matrix_free:
                    out = gmres(self.kernel, rhs, tol = self.tol, maxiter=self.maxiter)[0]
                    residual = cp.linalg.norm(rhs - self.kernel@out)/cp.linalg.norm(rhs)
                else:
                    self.K = self.kernel.construct(self.kernel.rho)
                    out = gmres(self.K, rhs, tol = self.tol, maxiter=self.maxiter)[0]
                    residual = cp.linalg.norm(rhs - self.K@out)/cp.linalg.norm(rhs)
        else:
            raise ValueError("Solver requires a density vector to be passed or set on the kernel.")
        
        return out, residual
    
class SPSOLVE(Solver):
    def __init__(self, kernel: CuStiffnessKernel):
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
        
        out = cp.copy(rhs)
        
        if self.kernel.has_con:
            K = K[:,self.kernel.non_con_map][self.kernel.non_con_map,:]
            rhs_ = rhs[self.kernel.non_con_map]
        
        
        out[self.kernel.non_con_map] = spsolve(K, rhs_)
        
        residual = cp.linalg.norm(rhs - self.kernel@out)/cp.linalg.norm(rhs)
        
        return out, residual