import numpy as np
from ._base import Solver
from ..kernels._processors import StiffnessKernel, GeneralStiffnessKernel, UniformStiffnessKernel
from scipy.sparse.linalg import gmres as sp_gmres, cg as sp_cg, bicgstab as sp_bicgstab
from scipy.linalg import get_lapack_funcs
from scipy.sparse.linalg import aslinearoperator
import logging
logger = logging.getLogger(__name__)

def cg(A, b, x0=None, rtol=1e-5, maxiter=1000):
    tol = rtol
    normb = (b*b).sum()**0.5
    
    if x0 is None:
        x = b.copy()
    else:
        x = x0
    r = b - A.matvec(x)
    
    rho_prev, p = None, None
    
    for iteration in range(maxiter):
        if (r*r).sum()**0.5/normb < tol:
            return x, iteration
        z = r
        rho_cur = (r*z).sum()
        if iteration > 0:
            beta = rho_cur / rho_prev
            p *= beta
            p += z
        else:  # First spin
            p = np.empty_like(r)
            p[:] = z[:]

        q = A.matvec(p)
        alpha = rho_cur / (p*q).sum()
        x += alpha*p
        r -= alpha*q
        rho_prev = rho_cur

    return x, iteration

def bicgstab(A, b, x0=None, rtol=1e-5, maxiter=1000):
    rhotol = np.finfo(x.dtype.char).eps**2
    omegatol = rhotol
    matvec = A.matvec
    
    tol = rtol
    normb = (b*b).sum()**0.5
    
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0
        
    # Dummy values to initialize vars, silence linter warnings
    rho_prev, omega, alpha, p, v = None, None, None, None, None

    r = b - matvec(x) if x.any() else b.copy()
    rtilde = r.copy()

    for iteration in range(maxiter):
        if (r*r).sum()**0.5/normb < tol:  # Are we done?
            return x, 0

        rho = (rtilde*r).sum()
        if np.abs(rho) < rhotol:  # rho breakdown
            return x, -10

        if iteration > 0:
            if np.abs(omega) < omegatol:  # omega breakdown
                return x, -11

            beta = (rho / rho_prev) * (alpha / omega)
            p -= omega*v
            p *= beta
            p += r
        else:  # First spin
            s = np.empty_like(r)
            p = r.copy()

        phat = p
        v = matvec(phat)
        rv = (rtilde* v).sum()
        if rv == 0:
            return x, -11
        alpha = rho / rv
        r -= alpha*v
        s[:] = r[:]

        if (s*s).sum()**0.5/normb < tol:
            x += alpha*phat
            return x, 0

        shat = s
        t = matvec(shat)
        omega = (t*s).sum() / (t*t).sum()
        x += alpha*phat
        x += omega*shat
        r -= omega*t
        rho_prev = rho

    return x, maxiter

def gmres(A, b, x0=None, rtol=1e-5, maxiter=1000):
    
    Mb_nrm2 = (b*b).sum()**0.5
    bnrm2 = Mb_nrm2
    n = len(b)
    # ====================================================
    # =========== Tolerance control from gh-8400 =========
    # ====================================================
    # Tolerance passed to GMRESREVCOM applies to the inner
    # iteration and deals with the left-preconditioned
    # residual.
    ptol_max_factor = 1.
    ptol = Mb_nrm2 * min(ptol_max_factor, rtol / bnrm2)
    presid = 0.
    # ====================================================
    lartg = get_lapack_funcs('lartg', dtype=x.dtype)

    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0
    eps = np.finfo(x.dtype.char).eps
    restart = 20
    # allocate internal variables
    v = np.empty([restart+1, n], dtype=x.dtype)
    h = np.zeros([restart, restart+1], dtype=x.dtype)
    givens = np.zeros([restart, 2], dtype=x.dtype)

    # legacy iteration count
    inner_iter = 0

    for iteration in range(maxiter):
        if iteration == 0:
            r = b - A.matvec(x)
            if (r*r).sum()**0.5/bnrm2 < rtol:
                return x, 0

        v[0, :] = r
        tmp = (v[0, :]*v[0, :]).sum()**0.5
        v[0, :] *= (1 / tmp)
        # RHS of the Hessenberg problem
        S = np.zeros(restart+1, dtype=x.dtype)
        S[0] = tmp

        breakdown = False
        for col in range(restart):
            av = A.matvec(v[col, :])
            w = av

            # Modified Gram-Schmidt
            h0 = (w*w).sum()**0.5
            for k in range(col+1):
                tmp = (v[k, :]* w).sum()
                h[col, k] = tmp
                w -= tmp*v[k, :]

            h1 = (w*w).sum()**0.5
            h[col, col + 1] = h1
            v[col + 1, :] = w[:]

            # Exact solution indicator
            if h1 <= eps*h0:
                h[col, col + 1] = 0
                breakdown = True
            else:
                v[col + 1, :] *= (1 / h1)

            # apply past Givens rotations to current h column
            for k in range(col):
                c, s = givens[k, 0], givens[k, 1]
                n0, n1 = h[col, [k, k+1]]
                h[col, [k, k + 1]] = [c*n0 + s*n1, -s.conj()*n0 + c*n1]

            # get and apply current rotation to h and S
            c, s, mag = lartg(h[col, col], h[col, col+1])
            givens[col, :] = [c, s]
            h[col, [col, col+1]] = mag, 0

            # S[col+1] component is always 0
            tmp = -np.conjugate(s)*S[col]
            S[[col, col + 1]] = [c*S[col], tmp]
            presid = np.abs(tmp)
            inner_iter += 1

            if presid <= ptol or breakdown:
                break

        # Solve h(col, col) upper triangular system and allow pseudo-solve
        # singular cases as in (but without the f2py copies):
        # y = trsv(h[:col+1, :col+1].T, S[:col+1])

        if h[col, col] == 0:
            S[col] = 0

        y = np.zeros([col+1], dtype=x.dtype)
        y[:] = S[:col+1]
        for k in range(col, 0, -1):
            if y[k] != 0:
                y[k] /= h[k, k]
                tmp = y[k]
                y[:k] -= tmp*h[k, :k]
        if y[0] != 0:
            y[0] /= h[0, 0]

        x += y @ v[:col+1, :]

        r = b - A.matvec(x)
        rnorm = (r*r).sum()**0.5

        if rnorm/bnrm2 <= rtol:
            break
        elif breakdown:
            # Reached breakdown (= exact solution), but the external
            # tolerance check failed. Bail out with failure.
            break
        elif presid <= ptol:
            # Inner loop passed but outer didn't
            ptol_max_factor = max(eps, 0.25 * ptol_max_factor)
        else:
            ptol_max_factor = min(1.0, 1.5 * ptol_max_factor)

        ptol = presid * min(ptol_max_factor, rtol)

    info = 0 if (rnorm/bnrm2 <= rtol) else maxiter
    return x, info

class CG(Solver):
    def __init__(self, kernel: StiffnessKernel, maxiter=1000, tol=1e-5, matrix_free=None):
        super().__init__()
        self.kernel = kernel
        self.last_x0 = None
        self.tol = tol
        self.maxiter = maxiter
        
        if isinstance(kernel, GeneralStiffnessKernel) or isinstance(kernel, UniformStiffnessKernel):
            if matrix_free is None:
                matrix_free = False
                logger.warning("Matrix free is set to False for general and uniform kernels. This is recommended for speed. Will use more memory.")
            elif matrix_free:
                logger.warning("Matrix free is set to True for general and uniform kernels. This is not recommended for speed. Will use less memory however.")
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
                    out = cg(self.kernel, rhs, x0=self.last_x0, rtol = self.tol, maxiter=self.maxiter)[0]
                else:
                    self.K = aslinearoperator(self.kernel.construct(self.kernel.rho))
                    out = cg(self.K, rhs, x0=self.last_x0, rtol = self.tol, maxiter=self.maxiter)[0]
                self.last_x0 = out
            else:
                if self.matrix_free:
                    out = cg(self.kernel, rhs, rtol = self.tol, maxiter=self.maxiter)[0]
                else:
                    self.K = aslinearoperator(self.kernel.construct(self.kernel.rho))
                    out = cg(self.K, rhs, rtol = self.tol, maxiter=self.maxiter)[0]
        else:
            raise ValueError("Solver requires a density vector to be passed or set on the kernel.")
        
        r = rhs - self.kernel@out
        residual = (r*r).sum()**0.5/(rhs*rhs).sum()**0.5
        
        return out, residual
    
class BiCGSTAB(Solver):
    def __init__(self, kernel: StiffnessKernel, maxiter=1000, tol=1e-5, matrix_free=None):
        super().__init__()
        self.kernel = kernel
        self.last_x0 = None
        self.tol = tol
        self.maxiter = maxiter
        
        if isinstance(kernel, GeneralStiffnessKernel) or isinstance(kernel, UniformStiffnessKernel):
            if matrix_free is None:
                matrix_free = False
                logger.warning("Matrix free is set to False for general and uniform kernels. This is recommended for speed. Will use more memory.")
            elif matrix_free:
                logger.warning("Matrix free is set to True for general and uniform kernels. This is not recommended for speed. Will use less memory however.")
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
                    out = bicgstab(self.kernel, rhs, x0=self.last_x0, rtol = self.tol, maxiter=self.maxiter)[0]
                else:
                    self.K = self.kernel.construct(self.kernel.rho)
                    out = sp_bicgstab(self.K, rhs, x0=self.last_x0, rtol = self.tol, maxiter=self.maxiter)[0]
                self.last_x0 = out
            else:
                if self.matrix_free:
                    out = bicgstab(self.kernel, rhs, rtol = self.tol, maxiter=self.maxiter)[0]
                else:
                    self.K = self.kernel.construct(self.kernel.rho)
                    out = sp_bicgstab(self.K, rhs, rtol = self.tol, maxiter=self.maxiter)[0]
        else:
            raise ValueError("Solver requires a density vector to be passed or set on the kernel.")
        
        r = rhs - self.kernel@out
        residual = (r*r).sum()**0.5/(rhs*rhs).sum()**0.5
        
        return out, residual

class GMRES(Solver):
    def __init__(self, kernel: StiffnessKernel, maxiter=1000, tol=1e-5, matrix_free=None):
        super().__init__()
        self.kernel = kernel
        self.last_x0 = None
        self.tol = tol
        self.maxiter = maxiter
        
        if isinstance(kernel, GeneralStiffnessKernel) or isinstance(kernel, UniformStiffnessKernel):
            if matrix_free is None:
                matrix_free = False
                logger.warning("Matrix free is set to False for general and uniform kernels. This is recommended for speed. Will use more memory.")
            elif matrix_free:
                logger.warning("Matrix free is set to True for general and uniform kernels. This is not recommended for speed. Will use less memory however.")
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
                    out = gmres(self.kernel, rhs, rtol = self.tol, maxiter=self.maxiter)[0]
                else:
                    self.K = self.kernel.construct(self.kernel.rho)
                    out = sp_gmres(self.K, rhs, x0=self.last_x0, rtol = self.tol, maxiter=self.maxiter)[0]
                self.last_x0 = out
            else:
                if self.matrix_free:
                    out = gmres(self.kernel, rhs, rtol = self.tol, maxiter=self.maxiter)[0]
                else:
                    self.K = self.kernel.construct(self.kernel.rho)
                    out = sp_gmres(self.K, rhs, rtol = self.tol, maxiter=self.maxiter)[0]
        else:
            raise ValueError("Solver requires a density vector to be passed or set on the kernel.")
        
        residual = np.linalg.norm(rhs - self.kernel@out)/np.linalg.norm(rhs)
        
        self.last_x0 = out
        
        return out, residual