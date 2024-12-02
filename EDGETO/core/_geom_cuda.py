import cupy as cp
import numpy as np

from ._cuda import (
    generate_elements_2d_kernel,
    generate_elements_3d_kernel,
)

def generate_structured_mesh_cuda(dim, nel, dtype=cp.float64):
    """
    Generate structured mesh entirely on GPU using CUDA.
    
    Parameters:
        dim: Array with domain dimensions
        nel: Array with number of elements in each direction
        
    Returns:
        elements: Array of element connectivity (on GPU)
        node_positions: Array of nodal coordinates (on GPU)
    """
    # Convert inputs to GPU if needed
    dim = cp.asarray(dim)
    nel = cp.asarray(nel)
    
    if len(dim) != len(nel):
        raise ValueError("Dimensions of dim and nel must match")
        
    if len(dim) == 2:
        nx, ny = nel[0] + 1, nel[1] + 1
        L, H = dim[0], dim[1]
        
        # Generate elements
        num_elem = int((nx - 1) * (ny - 1))
        elements = cp.zeros((num_elem, 4), dtype=cp.int32)
        
        threads_per_block = 256
        blocks_per_grid = (num_elem + threads_per_block - 1) // threads_per_block
        generate_elements_2d_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (elements, int(nx), int(ny))
        )
        
        # Generate node positions using CuPy
        x = cp.linspace(0, L, int(nx), dtype=dtype)
        y = cp.linspace(0, H, int(ny), dtype=dtype)
        xx, yy = cp.meshgrid(x, y, copy=False)
        node_positions = cp.stack([xx.flatten(), yy.flatten()], axis=-1, dtype=dtype)
        
    elif len(dim) == 3:
        nx, ny, nz = nel[0] + 1, nel[1] + 1, nel[2] + 1
        L, H, W = dim[0], dim[1], dim[2]
        
        # Generate elements
        num_elem = int((nx - 1) * (ny - 1) * (nz - 1))
        elements = cp.zeros((num_elem, 8), dtype=cp.int32)
        
        threads_per_block = 256
        blocks_per_grid = (num_elem + threads_per_block - 1) // threads_per_block
        generate_elements_3d_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (elements, int(nx), int(ny), int(nz))
        )
        
        # Generate node positions using CuPy
        x = cp.linspace(0, L, int(nx), dtype=dtype)
        y = cp.linspace(0, H, int(ny), dtype=dtype)
        z = cp.linspace(0, W, int(nz), dtype=dtype)
        xx, yy, zz = cp.meshgrid(x, y, z, copy=False)
        node_positions = cp.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1, dtype=dtype)
        
    else:
        raise ValueError("Only 2D and 3D meshes are supported")
        
    return elements, node_positions