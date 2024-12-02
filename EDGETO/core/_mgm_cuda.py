import numpy as np
import cupy as cp
from ._cuda import (
    restriction_3d_kernel,
    restriction_2d_kernel,
    prolongation_3d_kernel,
    prolongation_2d_kernel,
    prolongation_matrix_2d_kernel,
    prolongation_matrix_3d_kernel,
    restriction_matrix_2d_kernel,
    restriction_matrix_3d_kernel,
    get_restricted_3d_l0_nnz_based,
    get_restricted_3d_l1p_nnz_based,
    get_restricted_2d_l0_nnz_based,
    get_restricted_2d_l1p_nnz_based,
    SOR_l0,
    SOR_l1p,
    csr_diagonal
)

from ..geom._mesh import CuStructuredMesh2D, CuStructuredMesh3D
from ..kernels._cuda import StructuredStiffnessKernel as CuStructuredStiffnessKernel
from typing import Union

def SOR_l0_cuda(kernel: Union[CuStructuredStiffnessKernel, CuStructuredStiffnessKernel], x, b, D_inv, omega, n_iter):
    n_tasks = kernel.shape[0]//kernel.dof
    
    threads_per_block = 256
    blocks_per_grid = (n_tasks + threads_per_block - 1) // threads_per_block
    
    for _ in range(n_iter):
        SOR_l0((blocks_per_grid,), (threads_per_block,),
                (kernel.K_single,
                kernel.elements_flat,
                kernel.el_ids,
                kernel.rho,
                kernel.sorter,
                kernel.node_ids,
                kernel.n_nodes,
                x,
                b,
                D_inv,
                kernel.dof,
                kernel.elements_size,
                kernel.elements_flat.shape[0],
                kernel.constraints,
                omega)
        )
    
def SOR_l1p_cuda(A: cp.sparse.csr_matrix, x, b, D_inv, omega, n_iter, dof):
    n_tasks = A.shape[0]//dof
    
    threads_per_block = 256
    blocks_per_grid = (n_tasks + threads_per_block - 1) // threads_per_block
    
    for _ in range(n_iter):
        SOR_l1p((blocks_per_grid,), (threads_per_block,),
            (A.data,
             A.indptr,
             A.indices,
             x,
             b,
             D_inv,
             dof,
             omega,
             n_tasks)
        )

def get_restricted_l0_cuda(mesh: Union[CuStructuredMesh2D, CuStructuredMesh3D], kernel: CuStructuredStiffnessKernel, Cp = None):
    
    dim = mesh.nodes.shape[1]

    if dim == 2:
        nx_fine = mesh.nelx+1
        ny_fine = mesh.nely+1
        nx_coarse = mesh.nelx // 2 + 1
        ny_coarse = mesh.nely // 2 + 1
        dof = mesh.dof
        node_ids = kernel.node_ids
        el_ids = kernel.el_ids
        sorter = kernel.sorter
        elements_flat = kernel.elements_flat
        elements_size = kernel.elements_size
        con_map = kernel.constraints
        elem_flat_size = elements_flat.shape[0]
        n_nodes = kernel.n_nodes
        
        n_tasks = nx_coarse * ny_coarse * 9
        
        threads_per_block = 256
        blocks_per_grid = (n_tasks + threads_per_block - 1) // threads_per_block
        
        if Cp is None:
            nnz = cp.ones(n_tasks//9*dof, dtype=cp.int32) * 18
            Cp = cp.zeros(n_tasks//9*dof + 1, dtype=cp.int32)
            Cp[1:] = cp.cumsum(nnz)

        K_flat = kernel.K_single
        Cj = cp.zeros(int(Cp[-1]), dtype=np.int32)
        Cx = cp.zeros(int(Cp[-1]), dtype=K_flat.dtype)
        weights = kernel.rho
        
        get_restricted_2d_l0_nnz_based((blocks_per_grid,),
                                        (threads_per_block,),
                                        (K_flat,
                                         nx_fine, ny_fine, nx_coarse, ny_coarse,
                                         Cp, Cj, Cx,
                                         dof, node_ids, el_ids, sorter, elements_flat, weights, elements_size, con_map, elem_flat_size, n_nodes))
    
    else:
        nx_fine = mesh.nelx+1
        ny_fine = mesh.nely+1
        nz_fine = mesh.nelz+1
        nx_coarse = mesh.nelx // 2 + 1
        ny_coarse = mesh.nely // 2 + 1
        nz_coarse = mesh.nelz // 2 + 1
        dof = mesh.dof
        node_ids = kernel.node_ids
        el_ids = kernel.el_ids
        sorter = kernel.sorter
        elements_flat = kernel.elements_flat
        elements_size = kernel.elements_size
        con_map = kernel.constraints
        elem_flat_size = elements_flat.shape[0]
        n_nodes = kernel.n_nodes
        
        n_tasks = nx_coarse * ny_coarse * nz_coarse * 27

        threads_per_block = 256
        blocks_per_grid = (n_tasks + threads_per_block - 1) // threads_per_block
        
        if Cp is None:
            nnz = cp.ones(n_tasks//27*dof, dtype=cp.int32) * 81
            Cp = cp.zeros(n_tasks//27*dof + 1, dtype=cp.int32)
            Cp[1:] = cp.cumsum(nnz)
        
        K_flat = kernel.K_single
        Cj = cp.zeros(int(Cp[-1]), dtype=np.int32)
        Cx = cp.zeros(int(Cp[-1]), dtype=K_flat.dtype)
        weights = kernel.rho
        get_restricted_3d_l0_nnz_based((blocks_per_grid,), 
                                       (threads_per_block,), 
                                       (K_flat,
                                        nx_fine, ny_fine, nz_fine, nx_coarse, ny_coarse, nz_coarse,
                                        Cp, Cj, Cx,
                                        dof, node_ids, el_ids, sorter, elements_flat, weights, elements_size, con_map, elem_flat_size, n_nodes))    
    
    K = cp.sparse.csr_matrix((Cx, Cj, Cp), shape=(Cp.shape[0]-1, Cp.shape[0]-1))
    K.has_canonical_format = True
    diag = cp.zeros(K.shape[0], dtype=K.dtype)
    n_row = K.shape[0]
    threads_per_block = 256
    blocks_per_grid = (n_row + threads_per_block - 1) // threads_per_block
    
    csr_diagonal((blocks_per_grid,), (threads_per_block,), (K.data, K.indptr, K.indices, diag, n_row))
    
    K.diagonal = lambda: diag
    
    return K

def get_restricted_l1p_cuda(A : cp.sparse.csr_matrix, nel, dof, Cp = None):
    
    dim = len(nel)
    if dim == 2:
        nx_fine = int(nel[0] + 1)
        ny_fine = int(nel[1] + 1)
        nx_coarse = int(nel[0] // 2 + 1)
        ny_coarse = int(nel[1] // 2 + 1)
        
        n_tasks = nx_coarse*ny_coarse*9
        
        threads_per_block = 256
        blocks_per_grid = (n_tasks + threads_per_block - 1) // threads_per_block
        
        if Cp is None:
            nnz = cp.ones(n_tasks//9*dof, dtype=cp.int32) * 18
            Cp = cp.zeros(n_tasks//9*dof + 1, dtype=cp.int32)
            Cp[1:] = cp.cumsum(nnz)
        
        Cj = cp.zeros(int(Cp[-1]), dtype=np.int32)
        Cx = cp.zeros(int(Cp[-1]), dtype=A.dtype)
        
        get_restricted_2d_l1p_nnz_based((blocks_per_grid,), (threads_per_block,), 
                              (A.data, A.indices, A.indptr, nx_fine, ny_fine, nx_coarse, ny_coarse, Cp, Cj, Cx, dof))
                
    else:
        nx_fine = int(nel[0] + 1)
        ny_fine = int(nel[1] + 1)
        nz_fine = int(nel[2] + 1)
        nx_coarse = int(nel[0] // 2 + 1)
        ny_coarse = int(nel[1] // 2 + 1)
        nz_coarse = int(nel[2] // 2 + 1)
        
        n_tasks = nx_coarse*ny_coarse*nz_coarse*27
        
        threads_per_block = 256
        blocks_per_grid = (n_tasks + threads_per_block - 1) // threads_per_block
        
        if Cp is None:
            nnz = cp.ones(n_tasks//27 * dof, dtype=np.int32) * 81
            Cp = cp.zeros(n_tasks//27 * dof + 1, dtype=np.int32)
            Cp[1:] = cp.cumsum(nnz)

        
        Cj = cp.zeros(int(Cp[-1]), dtype=np.int32)
        Cx = cp.zeros(int(Cp[-1]), dtype=A.dtype)

        get_restricted_3d_l1p_nnz_based((blocks_per_grid,), (threads_per_block,),
                              (A.data, A.indices, A.indptr, nx_fine, ny_fine, nz_fine, nx_coarse, ny_coarse, nz_coarse, Cp, Cj, Cx, dof))
                
    K = cp.sparse.csr_matrix((Cx, Cj, Cp), shape=(Cp.shape[0]-1, Cp.shape[0]-1))
    # K.has_canonical_format = True
    # diag = cp.zeros(K.shape[0], dtype=K.dtype)
    # n_row = K.shape[0]
    # threads_per_block = 256
    # blocks_per_grid = (n_row + threads_per_block - 1) // threads_per_block
    
    # csr_diagonal((blocks_per_grid,), (threads_per_block,), (K.data, K.indptr, K.indices, diag, n_row))
    
    # K.diagonal = lambda: diag
    return K

def apply_restriction_cuda(v, nel, dof):

    # Convert input to GPU if needed
    if isinstance(v, np.ndarray):
        v = cp.array(v, dtype=v.dtype)
    
    if len(nel) == 3:
        nx_fine, ny_fine, nz_fine = nel[0] + 1, nel[1] + 1, nel[2] + 1
        nel_coarse = (nel[0] // 2, nel[1] // 2, nel[2] // 2)
        nx_coarse, ny_coarse, nz_coarse = nel_coarse[0] + 1, nel_coarse[1] + 1, nel_coarse[2] + 1
        
        # Allocate output array
        n_coarse_nodes = nx_coarse * ny_coarse * nz_coarse
        v_coarse = cp.zeros(n_coarse_nodes * dof, dtype=v.dtype)
        
        # Configure kernel launch parameters
        threads_per_block = 256
        blocks_per_grid = (n_coarse_nodes + threads_per_block - 1) // threads_per_block
        
        # Launch kernel
        restriction_3d_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (v, v_coarse, nx_fine, ny_fine, nz_fine,
             nx_coarse, ny_coarse, nz_coarse, dof)
        )
        
    else:  # 2D case
        nx_fine, ny_fine = nel[0] + 1, nel[1] + 1
        nel_coarse = (nel[0] // 2, nel[1] // 2)
        nx_coarse, ny_coarse = nel_coarse[0] + 1, nel_coarse[1] + 1
        
        # Allocate output array
        n_coarse_nodes = nx_coarse * ny_coarse
        v_coarse = cp.zeros(n_coarse_nodes * dof, dtype=v.dtype)
        
        # Configure kernel launch parameters
        threads_per_block = 256
        blocks_per_grid = (n_coarse_nodes + threads_per_block - 1) // threads_per_block
        
        # Launch kernel
        restriction_2d_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (v, v_coarse, nx_fine, ny_fine, nx_coarse, ny_coarse, dof)
        )
    
    return v_coarse

def apply_prolongation_cuda(v, nel, dof):
    # Convert input to GPU if needed
    if isinstance(v, np.ndarray):
        v = cp.array(v, dtype=v.dtype)
    
    if len(nel) == 3:
        nx_fine, ny_fine, nz_fine = nel[0] + 1, nel[1] + 1, nel[2] + 1
        nel_coarse = (nel[0] // 2, nel[1] // 2, nel[2] // 2)
        nx_coarse, ny_coarse, nz_coarse = nel_coarse[0] + 1, nel_coarse[1] + 1, nel_coarse[2] + 1
        
        # Allocate output array
        n_fine_nodes = nx_fine * ny_fine * nz_fine
        v_fine = cp.zeros(n_fine_nodes * dof, dtype=v.dtype)
        
        # Configure kernel launch parameters
        threads_per_block = 256
        blocks_per_grid = (n_fine_nodes + threads_per_block - 1) // threads_per_block
        
        # Launch kernel
        prolongation_3d_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (v, v_fine, nx_fine, ny_fine, nz_fine,
             nx_coarse, ny_coarse, nz_coarse, dof)
        )
        
    else:  # 2D case
        nx_fine, ny_fine = nel[0] + 1, nel[1] + 1
        nel_coarse = (nel[0] // 2, nel[1] // 2)
        nx_coarse, ny_coarse = nel_coarse[0] + 1, nel_coarse[1] + 1
        
        # Allocate output array
        n_fine_nodes = nx_fine * ny_fine
        v_fine = cp.zeros(n_fine_nodes * dof, dtype=v.dtype)
        
        # Configure kernel launch parameters
        threads_per_block = 256
        blocks_per_grid = (n_fine_nodes + threads_per_block - 1) // threads_per_block
        
        # Launch kernel
        prolongation_2d_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (v, v_fine, nx_fine, ny_fine, nx_coarse, ny_coarse, dof)
        )
    
    return v_fine

def get_restriction_matrix_cuda(nel, dof, dtype=cp.float64):
    if len(nel) == 2:
        nx_fine, ny_fine = nel[0] + 1, nel[1] + 1
        nx_coarse = nel[0] // 2 + 1
        ny_coarse = nel[1] // 2 + 1
        n_coarse_nodes = nx_coarse * ny_coarse
        n_fine_nodes = nx_fine * ny_fine
        # Max number of neighbors in 2D stencil is 9
        nnz_per_row = cp.ones(n_coarse_nodes * dof, dtype=cp.int32) * 9
    else:
        nx_fine, ny_fine, nz_fine = nel[0] + 1, nel[1] + 1, nel[2] + 1
        nx_coarse = nel[0] // 2 + 1
        ny_coarse = nel[1] // 2 + 1
        nz_coarse = nel[2] // 2 + 1
        n_coarse_nodes = nx_coarse * ny_coarse * nz_coarse
        n_fine_nodes = nx_fine * ny_fine * nz_fine
        # Max number of neighbors in 3D stencil is 27
        nnz_per_row = cp.ones(n_coarse_nodes * dof, dtype=cp.int32) * 27
    
    Cp = cp.zeros(n_coarse_nodes * dof + 1, dtype=cp.int32)
    Cp[1:] = cp.cumsum(nnz_per_row)
    Cj = cp.zeros(int(Cp[-1]), dtype=cp.int32)
    Cx = cp.zeros(int(Cp[-1]), dtype=dtype)
    
    # Launch parameters
    threadsperblock = 256
    blockspergrid = (n_coarse_nodes + (threadsperblock - 1)) // threadsperblock
    
    if len(nel) == 2:
        restriction_matrix_2d_kernel((blockspergrid,), (threadsperblock,),
                                   (nel[0], nel[1], dof, Cp, Cj, Cx))
    else:
        restriction_matrix_3d_kernel((blockspergrid,), (threadsperblock,),
                                   (nel[0], nel[1], nel[2], dof, Cp, Cj, Cx))
    
    return cp.sparse.csr_matrix((Cx, Cj, Cp), shape=(n_coarse_nodes * dof, n_fine_nodes * dof))

def get_prolongation_matrix_cuda(nel, dof, dtype=cp.float64):
    if len(nel) == 2:
        nx_fine, ny_fine = nel[0] + 1, nel[1] + 1
        nx_coarse = nel[0] // 2 + 1
        ny_coarse = nel[1] // 2 + 1
        n_fine_nodes = nx_fine * ny_fine
        n_coarse_nodes = nx_coarse * ny_coarse
        # Max number of coarse nodes contributing to a fine node in 2D is 4
        nnz_per_row = cp.ones(n_fine_nodes * dof, dtype=cp.int32) * 4
    else:
        nx_fine, ny_fine, nz_fine = nel[0] + 1, nel[1] + 1, nel[2] + 1
        nx_coarse = nel[0] // 2 + 1
        ny_coarse = nel[1] // 2 + 1
        nz_coarse = nel[2] // 2 + 1
        n_fine_nodes = nx_fine * ny_fine * nz_fine
        n_coarse_nodes = nx_coarse * ny_coarse * nz_coarse
        # Max number of coarse nodes contributing to a fine node in 3D is 8
        nnz_per_row = cp.ones(n_fine_nodes * dof, dtype=cp.int32) * 8
    
    Cp = cp.zeros(n_fine_nodes * dof + 1, dtype=cp.int32)
    Cp[1:] = cp.cumsum(nnz_per_row)
    Cj = cp.zeros(int(Cp[-1]), dtype=cp.int32)
    Cx = cp.zeros(int(Cp[-1]), dtype=dtype)
    
    # Launch parameters
    threadsperblock = 256
    blockspergrid = (n_fine_nodes + (threadsperblock - 1)) // threadsperblock
    
    if len(nel) == 2:
        prolongation_matrix_2d_kernel((blockspergrid,), (threadsperblock,),
                                    (nel[0], nel[1], dof, Cp, Cj, Cx))
    else:
        prolongation_matrix_3d_kernel((blockspergrid,), (threadsperblock,),
                                    (nel[0], nel[1], nel[2], dof, Cp, Cj, Cx))
    
    return cp.sparse.csr_matrix((Cx, Cj, Cp), shape=(n_fine_nodes * dof, n_coarse_nodes * dof))