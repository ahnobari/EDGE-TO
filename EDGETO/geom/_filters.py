from ..core._filter_cuda import (apply_filter_3D_cuda,
                                 apply_filter_2D_cuda,
                                 get_filter_weights_2D_cuda,
                                 get_filter_weights_3D_cuda,
                                 apply_filter_2D_transpose_cuda,
                                 apply_filter_3D_transpose_cuda)

from ..core._filter import (apply_filter_2D_parallel,
                            apply_filter_3D_parallel,
                            filter_kernel_2D_general,
                            filter_kernel_3D_general,
                            get_filter_weights_2D,
                            get_filter_weights_3D,
                            apply_filter_2D_parallel_transpose,
                            apply_filter_3D_parallel_transpose,
                            )
from ._mesh import StructuredMesh2D, StructuredMesh3D, CuStructuredMesh2D, CuStructuredMesh3D, GeneralMesh, CuGeneralMesh
import numpy as np
import cupy as cp

class _TransposeView:
    def __init__(self, original):
        self._original = original
    
    def __matmul__(self, rhs):
        return self._original._rmatvec(rhs)
    
    def dot(self, rhs):
        return self._original._rmatvec(rhs)
    
    @property
    def T(self):
        return self._original

    def getattr(self, name):
        # Delegate any other attribute access to original
        return getattr(self._original, name)
    
class FilterKernel:
    def __init__(self):
        self.weights = None
        self.offsets = None
        self.shape = None
        self.matvec = self.dot
        
    def _matvec(self, rho):
        pass
    
    def _rmatvec(self, rho):
        pass
    
    def dot(self, rho):
        if isinstance(rho, type(self.weights)):
            if rho.ndim == 1:
                if rho.shape[0] == self.shape[1]:
                    return self._matvec(rho)
                else:
                    raise ValueError("Input vector size does not match the filter kernel size.")
            else:
                raise NotImplementedError("Only vector inputs are supported.")
        else:
            raise ValueError(f"Input must be a {type(self.weights)} array vector.")
    
    def __matmul__(self, rhs):
        return self.dot(rhs)
    
    @property
    def T(self):
        return _TransposeView(self)
        
    
class StructuredFilter3D(FilterKernel):
    def __init__(self, mesh: StructuredMesh3D, r_min):
        super().__init__()
        self.dtype = mesh.dtype

        self.nelx = mesh.nelx
        self.nely = mesh.nely
        self.nelz = mesh.nelz
        self.r_min = r_min
        self.shape = (self.nelx * self.nely * self.nelz, self.nelx * self.nely * self.nelz)
        dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
        self.scales = np.array([dx, dy, dz], dtype=self.dtype)
        self.scales = self.scales / self.scales.min()
        
        filter_rad = r_min
        n_neighbours = int(np.ceil(filter_rad))
        offset_range = np.arange(-n_neighbours, n_neighbours + 1, dtype=np.int32)
        
        a, b, c = np.meshgrid(offset_range, offset_range, offset_range, indexing='ij')
        offsets = np.vstack([a.ravel(), b.ravel(), c.ravel()]).T.astype(np.int32)
        offsets_adjusted = offsets * self.scales[None]
        
        distances = np.linalg.norm(offsets_adjusted, axis=1)
        weights = (r_min - distances) / r_min
        valid_mask = weights > 0
        offsets = offsets[valid_mask]
        weights = weights[valid_mask]
        weights /= weights.sum()
        weights = weights.astype(self.dtype)
        
        self.weights = weights
        self.offsets = offsets
        
        self.normalizer = get_filter_weights_3D(self.nelx, self.nely, self.nelz, self.offsets, self.weights)
        
    def _matvec(self, rho):
        v_out = np.zeros_like(rho)
        apply_filter_3D_parallel(rho, v_out, self.nelx, self.nely, self.nelz, self.offsets, self.weights)
        return v_out
    
    def _rmatvec(self, rho):
        v_out = np.zeros_like(rho)
        apply_filter_3D_parallel_transpose(rho, v_out, self.nelx, self.nely, self.nelz, self.offsets, self.weights, self.normalizer)
        return v_out


class StructuredFilter2D(FilterKernel):
    def __init__(self, mesh: StructuredMesh2D, r_min):
        super().__init__()
        self.nelx = mesh.nel[0]
        self.nely = mesh.nel[1]
        self.r_min = r_min
        self.shape = (self.nelx * self.nely, self.nelx * self.nely)
        self.dtype = mesh.dtype
        
        dx, dy = mesh.dx, mesh.dy
        self.scales = np.array([dx, dy], dtype=self.dtype)
        self.scales = self.scales / self.scales.min()
        
        filter_rad = r_min
        n_neighbours = int(np.ceil(filter_rad))
        offset_range = np.arange(-n_neighbours, n_neighbours + 1, dtype=np.int32)
        
        a, b = np.meshgrid(offset_range, offset_range, indexing='ij')
        offsets = np.vstack([a.ravel(), b.ravel()]).T.astype(np.int32)
        offsets_adjusted = offsets * self.scales[None]
        
        distances = np.linalg.norm(offsets_adjusted, axis=1)
        weights = (r_min - distances) / r_min
        valid_mask = weights > 0
        offsets = offsets[valid_mask]
        weights = weights[valid_mask]
        weights /= weights.sum()
        weights = weights.astype(self.dtype)
        
        self.weights = weights
        self.offsets = offsets
        
        self.normalizer = get_filter_weights_2D(self.nelx, self.nely, self.offsets, self.weights)
        
        
    def _matvec(self, rho):
        v_out = np.zeros_like(rho)
        apply_filter_2D_parallel(rho, v_out, self.nelx, self.nely, self.offsets, self.weights)
        return v_out
    
    def _rmatvec(self, rho):
        v_out = np.zeros_like(rho)
        apply_filter_2D_parallel_transpose(rho, v_out, self.nelx, self.nely, self.offsets, self.weights, self.normalizer)
        return v_out


class CuStructuredFilter3D(FilterKernel):
    def __init__(self, mesh: CuStructuredMesh3D, r_min):
        super().__init__()
        self.nelx = mesh.nel[0]
        self.nely = mesh.nel[1]
        self.nelz = mesh.nel[2]
        self.r_min = r_min
        self.shape = (self.nelx * self.nely * self.nelz, self.nelx * self.nely * self.nelz)
        
        self.dtype = mesh.dtype
        
        dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
        self.scales = cp.array([dx, dy, dz], dtype=self.dtype)
        self.scales = self.scales / self.scales.min()
        
        filter_rad = r_min
        n_neighbours = int(np.ceil(filter_rad))
        offset_range = cp.arange(-n_neighbours, n_neighbours + 1, dtype=cp.int32)
        
        a, b, c = cp.meshgrid(offset_range, offset_range, offset_range, indexing='ij')
        offsets = cp.vstack([a.ravel(), b.ravel(), c.ravel()]).T
        offsets_adjusted = offsets * self.scales[None]
        
        distances = cp.linalg.norm(offsets_adjusted, axis=1)
        weights = (r_min - distances) / r_min
        valid_mask = weights > 0
        offsets = offsets[valid_mask]
        weights = weights[valid_mask]
        weights /= weights.sum()
        
        self.weights = cp.array(weights, dtype=self.dtype)
        self.offsets = cp.array(offsets, dtype=cp.int32)
        
        self.normalizer = cp.zeros(self.shape[0], dtype=self.dtype)
        get_filter_weights_3D_cuda(self.nelx, self.nely, self.nelz, self.offsets, self.weights, self.normalizer)

    def _matvec(self, rho):
        v_out = cp.zeros_like(rho)
        apply_filter_3D_cuda(rho, v_out, self.nelx, self.nely, self.nelz, self.offsets, self.weights)
        return v_out

    def _rmatvec(self, rho):
        v_out = cp.zeros_like(rho)
        apply_filter_3D_transpose_cuda(rho, v_out, self.nelx, self.nely, self.nelz, self.offsets, self.weights, self.normalizer)
        return v_out


class CuStructuredFilter2D(FilterKernel):
    def __init__(self, mesh: CuStructuredMesh2D, r_min):
        super().__init__()
        self.nelx = int(mesh.nelx)
        self.nely = int(mesh.nely)
        self.r_min = r_min
        self.shape = (self.nelx * self.nely, self.nelx * self.nely)
        self.dtype = mesh.dtype
        
        dx, dy = mesh.dx, mesh.dy
        self.scales = cp.array([dx, dy], dtype=self.dtype)
        self.scales = self.scales / self.scales.min()
        
        filter_rad = r_min
        n_neighbours = int(np.ceil(filter_rad))
        offset_range = cp.arange(-n_neighbours, n_neighbours + 1, dtype=cp.int32)
        
        a, b = cp.meshgrid(offset_range, offset_range, indexing='ij')
        offsets = cp.vstack([a.ravel(), b.ravel()]).T
        offsets_adjusted = offsets * self.scales[None]
        
        distances = cp.linalg.norm(offsets_adjusted, axis=1)
        weights = (r_min - distances) / r_min
        valid_mask = weights > 0
        offsets = offsets[valid_mask]
        weights = weights[valid_mask]
        weights /= weights.sum()
        
        self.weights = cp.array(weights, dtype=self.dtype)
        self.offsets = cp.array(offsets, dtype=cp.int32)
        
        self.normalizer = cp.zeros(self.shape[0], dtype=self.dtype)
        get_filter_weights_2D_cuda(self.nelx, self.nely, self.offsets, self.weights, self.normalizer)
    
    def _matvec(self, rho):
        v_out = cp.zeros_like(rho)
        apply_filter_2D_cuda(rho, v_out, self.nelx, self.nely, self.offsets, self.weights)
        return v_out
    
    def _rmatvec(self, rho):
        v_out = cp.zeros_like(rho)
        apply_filter_2D_transpose_cuda(rho, v_out, self.nelx, self.nely, self.offsets, self.weights, self.normalizer)
        return v_out


class GeneralFilter(FilterKernel):
    def __init__(self, mesh: GeneralMesh, r_min):
        super().__init__()
        self.dtype = mesh.dtype
        self.nd = mesh.nodes.shape[1]
        if self.nd == 2:
            self.kernel = filter_kernel_2D_general(mesh.elements, mesh.centeroids, r_min)
        else:
            self.kernel = filter_kernel_3D_general(mesh.elements, mesh.centeroids, r_min)
        
        self.kernel = self.kernel.astype(self.dtype)
        self.shape = self.kernel.shape 
        
        self.weights = np.empty(self.shape[0], dtype=self.dtype)
        
    def _matvec(self, rho):
        return (self.kernel @ rho).reshape(rho.shape)
    
    def _rmatvec(self, rho):
        return (self.kernel.T @ rho).reshape(rho.shape)


class CuGeneralFilter(FilterKernel):
    def __init__(self, mesh: CuGeneralMesh, r_min):
        super().__init__()
        self.dtype = mesh.dtype
        self.nd = mesh.nodes.shape[1]
        if self.nd == 2:
            self.kernel = filter_kernel_2D_general(mesh.elements, mesh.centeroids, r_min)
        else:
            self.kernel = filter_kernel_3D_general(mesh.elements, mesh.centeroids, r_min)
        
        self.kernel = cp.sparse.csr_matrix(self.kernel, dtype=self.dtype)
        
        self.shape = self.kernel.shape 
        
        self.weights = cp.empty(self.shape[0], dtype=self.dtype)
        
    def _matvec(self, rho):
        return (self.kernel @ rho).reshape(rho.shape)
    
    def _rmatvec(self, rho):
        return (self.kernel.T @ rho).reshape(rho.shape)
