import numpy as np
from ._base import Solver
from ..core._mgm import apply_restriction, apply_prolongation, get_restricted_l0, get_restricted_l1p
from ..kernels._processors import StiffnessKernel
from ..geom._mesh import StructuredMesh2D, StructuredMesh3D
from scipy.sparse.linalg import splu, spsolve
from ._iteratives import cg, bicgstab, gmres
from typing import Union


class MultiGrid(Solver):
    def __init__(self, mesh: Union[StructuredMesh2D,StructuredMesh3D],
                 kernel: StiffnessKernel, maxiter=1000, tol=1e-5, n_smooth=3,
                 omega=0.5 , n_level = 3, cycle='W', w_level=1, coarse_solver='splu',
                 matrix_free=False, low_level_tol = 1e-8, low_level_maxiter=5000, min_omega=0.4, omega_boost=1.06):
        super().__init__()
        self.kernel = kernel
        self.mesh = mesh
        self.last_x0 = None
        self.tol = tol
        self.maxiter = maxiter
        self.n_smooth = n_smooth
        self.omega = omega
        self.max_omega = omega
        self.min_omega = max(omega/2, min_omega)
        self.d_omega = (self.max_omega - self.min_omega)
        self.omega_boost = omega_boost
        self.n_level = n_level
        self.cycle = cycle
        self.w_level = w_level
        self.matrix_free = matrix_free
        self.dof = self.mesh.dof
        self.coarse_solver = coarse_solver
        self.low_level_tol = low_level_tol
        self.low_level_maxiter = low_level_maxiter
        
        if isinstance(self.w_level, int):
            self.w_level = [self.w_level]
        
        if self.coarse_solver in ['splu', 'spsolve'] and self.matrix_free:
            raise ValueError("Matrix free is not supported with splu solver use cg instead.")
        
        if self.coarse_solver not in ['cg', 'bicgstab', 'gmres', 'splu', 'spsolve']:
            raise ValueError("Coarse solver not recognized.")
        
        if not self.matrix_free:
            self.ptr = None
            self.PRs = []
        else:
            raise NotImplementedError("Matrix free is not implemented yet.")
            
    def reset(self):
        self.ptr = None
    
    def _jacobi_smoother(self, x, b, A, D_inv, n_step):
        for _ in range(n_step):
            x += self.omega * D_inv * (b - A @ x)
        return x
    
    def _setup(self):
        self.levels = []
        
        D = 1/self.kernel.diagonal()
        self.levels.append((self.kernel, D))
        
        for i in range(self.n_level):
            if i == 0 and not self.ptr is None:
                op = get_restricted_l0(self.mesh, self.kernel, Cp = self.ptr[0])
                D = 1/op.diagonal()
                self.levels.append((op, D))
                
            elif i == 0:
                self.ptr = []
                if self.mesh.nel.shape[0] == 2:
                    nnz = np.ones((self.mesh.nel[0]//2+1)*(self.mesh.nel[1]//2+1)*self.dof, dtype=np.int32) * 18
                    Cp = np.zeros((self.mesh.nel[0]//2+1)*(self.mesh.nel[1]//2+1)*self.dof + 1, dtype=np.int32)
                    Cp[1:] = np.cumsum(nnz)
                else:
                    nnz = np.ones((self.mesh.nel[0]//2+1)*(self.mesh.nel[1]//2+1)*(self.mesh.nel[2]//2+1)*self.dof, dtype=np.int32) * 81
                    Cp = np.zeros((self.mesh.nel[0]//2+1)*(self.mesh.nel[1]//2+1)*(self.mesh.nel[2]//2+1)*self.dof + 1, dtype=np.int32)
                    Cp[1:] = np.cumsum(nnz)
                
                self.ptr.append(np.copy(Cp))
                op = get_restricted_l0(self.mesh, self.kernel, Cp = Cp)
                D = 1/op.diagonal()
                self.levels.append((op, D))
                
            elif len(self.ptr) <= i:
                
                if self.mesh.nel.shape[0] == 2:
                    nnz = np.ones((self.mesh.nel[0]//(2**(i+1))+1)*(self.mesh.nel[1]//(2**(i+1))+1)*self.dof, dtype=np.int32) * 18
                    Cp = np.zeros((self.mesh.nel[0]//(2**(i+1))+1)*(self.mesh.nel[1]//(2**(i+1))+1)*self.dof + 1, dtype=np.int32)
                    Cp[1:] = np.cumsum(nnz)
                else:
                    nnz = np.ones((self.mesh.nel[0]//(2**(i+1))+1)*(self.mesh.nel[1]//(2**(i+1))+1)*(self.mesh.nel[2]//(2**(i+1))+1)*self.dof, dtype=np.int32) * 81
                    Cp = np.zeros((self.mesh.nel[0]//(2**(i+1))+1)*(self.mesh.nel[1]//(2**(i+1))+1)*(self.mesh.nel[2]//(2**(i+1))+1)*self.dof + 1, dtype=np.int32)
                    Cp[1:] = np.cumsum(nnz)
                
                self.ptr.append(np.copy(Cp))
                op = get_restricted_l1p(self.levels[-1][0], self.mesh.nel//(2**i), self.dof, Cp = Cp)
                D = 1/op.diagonal()
                self.levels.append((op, D))
            else:
                op = get_restricted_l1p(self.levels[-1][0], self.mesh.nel//(2**i), self.dof, Cp = self.ptr[i])
                D = 1/op.diagonal()
                self.levels.append((op, D))

    def _coarse_solver(self, K):
        if self.coarse_solver == 'cg':
            return lambda rhs: cg(K, rhs, rtol = self.low_level_tol, maxiter=self.low_level_maxiter)[0]
        elif self.coarse_solver == 'bicgstab':
            return lambda rhs: bicgstab(K, rhs, rtol = self.low_level_tol, maxiter=self.low_level_maxiter)[0]
        elif self.coarse_solver == 'gmres':
            return lambda rhs: gmres(K, rhs, rtol = self.low_level_tol, maxiter=self.low_level_maxiter)[0]
        elif self.coarse_solver == 'splu':
            SOLVER = splu(K)
            return lambda rhs: SOLVER.solve(rhs)
        elif self.coarse_solver == 'spsolve':
            return lambda rhs: spsolve(K, rhs)
        else:
            raise ValueError("Coarse solver not recognized.")
    
    def _multi_grid(self, x, b, level):
        if level == self.n_level:
            return self.coarse_solve(b)
        
        # presmooth
        A, D = self.levels[level]
        self.omega = self.omega * (1.06)**(level)
        x = self._jacobi_smoother(x, b, A, D, self.n_smooth)
        self.omega = self.omega / (1.06)**(level)
        # residual
        r = b - A@x
        
        nel = (self.mesh.nel // 2**level).astype(np.int32)
        
        # restrict
        coarse_residual = apply_restriction(r,nel,self.dof)
        # coarse_residual = self.PRs[level][1] @ r
        
        # go to next level
        coarse_u = coarse_residual*0.0
        e = self._multi_grid(coarse_u, coarse_residual, level+1)
        
        # prolongate
        e = apply_prolongation(e,nel,self.dof)
        # e = self.PRs[level][0] @ e
        
        self.omega = self.omega * (1.06)**(level)
        e = self._jacobi_smoother(e, r, A, D, self.n_smooth)
        self.omega = self.omega / (1.06)**(level)
        
        x += e
        
        if self.cycle == 'w' and level in self.w_level:
            r = b - A@x
        
            # restrict
            coarse_residual = apply_restriction(r,nel,self.dof)
            # coarse_residual = self.PRs[level][1] @ r
            
            # go to next level
            coarse_u = coarse_residual*0.0
            e = self._multi_grid(coarse_u, coarse_residual, level+1)
            
            # prolongate
            e = apply_prolongation(e,nel,self.dof)
            # e = self.PRs[level][0] @ e
            
            self.omega = self.omega * (1.06)**(level)
            e = self._jacobi_smoother(e, r, A, D, self.n_smooth)
            self.omega = self.omega / (1.06)**(level)
            
            x += e
        
        return x
    
    def solve(self, rhs, rho=None, use_last=True):
        
        if not (self.kernel.has_rho or (not rho is None)):
            raise ValueError("Solver requires a density vector to be passed or set on the kernel.")
        
        if not rho is None:
            rho = self.kernel.set_rho(rho)
        
        if use_last and self.last_x0 is not None:
            x = self.last_x0
        else:
            x = np.zeros_like(rhs)
            
        if self.matrix_free:
            self._mat_free_setup()
        else:
            self._setup()
        
        v_cycle = self._mat_free_multi_grid if self.matrix_free else self._multi_grid
        
        self.coarse_solve = self._coarse_solver(self.levels[-1][0])

        r = rhs - self.kernel.dot(x)
        z = v_cycle(np.zeros_like(r), r, 0)
        p = z.copy()
        rho_old = (r*z).sum()
        norm_b = (rhs*rhs).sum()**0.5
        
        for i in range(self.maxiter):
            q = self.kernel.dot(p)
            alpha = rho_old / (p*q).sum()
            x += alpha * p
            r -= alpha * q
            
            norm_r = (r*r).sum()**0.5
            R = norm_r / norm_b
            if R < self.tol:
                break
            self.omega = self.min_omega + self.d_omega/2*np.exp((-np.clip(R,self.tol,1e-1)+self.tol)*500)
            z = v_cycle(np.zeros_like(r), r,0)
            rho_new = (r*z).sum()
            beta = rho_new / rho_old
            p = z + beta * p
            rho_old = rho_new
        
        residual = norm_r / norm_b
        
        self.last_x0 = x
        
        del self.levels, self.coarse_solve
        
        
        return x, residual