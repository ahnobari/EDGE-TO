from ..core._geom import generate_structured_mesh
from ..core._geom_cuda import generate_structured_mesh_cuda
import numpy as np
import cupy as cp
from ..physics import auto_stiffness
import logging
logger = logging.getLogger(__name__)


class Mesh:
    pass


class StructuredMesh(Mesh):
    def __init__(self):
        pass


class StructuredMesh2D(StructuredMesh):
    def __init__(self, nx, ny, lx, ly, dtype=np.float64, stiffness=lambda x0: auto_stiffness(x0, 1.0, 0.33)):
        super().__init__()
        self.nelx = nx
        self.nely = ny
        self.lx = lx
        self.ly = ly
        self.nel = np.array([nx, ny], dtype=np.int32)
        self.dim = np.array([lx, ly], dtype=dtype)
        self.elements, self.nodes = generate_structured_mesh(self.dim,self.nel, dtype=dtype)
        self.elements_size = self.elements.shape[1]
        
        self.dx = lx / nx
        self.dy = ly / ny
        
        K, D, B, A = stiffness(self.nodes[self.elements[0]])
        self.K_single = K.astype(dtype)
        self.D_single = D.astype(dtype)
        self.B_single = B.astype(dtype)
        self.A_single = np.array([A], dtype=dtype)
        
        self.As = self.A_single
        self.volume = self.A_single[0] * self.nelx * self.nely

        self.dof = int(K.shape[0]/self.elements_size)
        
        self.dtype = dtype
        
        
class StructuredMesh3D(StructuredMesh):
    def __init__(self, nx, ny, nz, lx, ly, lz, dtype=np.float64, stiffness=lambda x0: auto_stiffness(x0, 1.0, 0.33)):
        super().__init__()
        self.nelx = nx
        self.nely = ny
        self.nelz = nz
        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.nel = np.array([nx, ny, nz], dtype=np.int32)
        self.dim = np.array([lx, ly, lz], dtype=dtype)
        self.elements, self.nodes = generate_structured_mesh(self.dim,self.nel, dtype=dtype)
        self.elements_size = self.elements.shape[1]
        
        self.dx = lx / nx
        self.dy = ly / ny
        self.dz = lz / nz
        
        K, D, B, A = stiffness(self.nodes[self.elements[0]])
        self.K_single = K.astype(dtype)
        self.D_single = D.astype(dtype)
        self.B_single = B.astype(dtype)
        self.A_single = np.array([A], dtype=dtype)
        
        self.As = self.A_single
        
        self.volume = self.A_single[0] * self.nelx * self.nely * self.nelz
        
        self.dof = int(K.shape[0]/self.elements_size)
        
        self.dtype = dtype


class CuStructuredMesh2D(StructuredMesh):
    def __init__(self, nx, ny, lx, ly, dtype=np.float64, stiffness=lambda x0: auto_stiffness(x0, 1.0, 0.33)):
        super().__init__()
        self.nelx = nx
        self.nely = ny
        self.lx = lx
        self.ly = ly
        self.nel = np.array([nx, ny], dtype=np.int32)
        self.dim = np.array([lx, ly], dtype=dtype)
        self.elements, self.nodes = generate_structured_mesh_cuda(self.dim,self.nel, dtype=dtype)
        self.elements_size = self.elements.shape[1]
        
        self.dx = lx / nx
        self.dy = ly / ny
        
        single_element = self.nodes[self.elements[0]].get()
        K, D, B, A = stiffness(single_element)
        self.K_single = cp.array(K, dtype=dtype)
        self.D_single = cp.array(D, dtype=dtype)
        self.B_single = cp.array(B, dtype=dtype)
        self.A_single = cp.array([A], dtype=dtype)
        
        self.As = self.A_single
        
        self.volume = self.A_single[0] * self.nelx * self.nely
        
        self.dof = int(K.shape[0]/self.elements_size)
        
        self.dtype = dtype


class CuStructuredMesh3D(StructuredMesh):
    def __init__(self, nx, ny, nz, lx, ly, lz, dtype=np.float64, stiffness=lambda x0: auto_stiffness(x0, 1.0, 0.33)):
        super().__init__()
        self.nelx = nx
        self.nely = ny
        self.nelz = nz
        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.nel = np.array([nx, ny, nz], dtype=np.int32)
        self.dim = np.array([lx, ly, lz], dtype=dtype)
        self.elements, self.nodes = generate_structured_mesh_cuda(self.dim,self.nel, dtype=dtype)
        self.elements_size = self.elements.shape[1]
        
        self.dx = lx / nx
        self.dy = ly / ny
        self.dz = lz / nz
        
        single_element = self.nodes[self.elements[0]].get()
        K, D, B, A = stiffness(single_element)
        self.K_single = cp.array(K, dtype=dtype)
        self.D_single = cp.array(D, dtype=dtype)
        self.B_single = cp.array(B, dtype=dtype)
        self.A_single = cp.array([A], dtype=dtype)
        
        self.As = self.A_single
        
        self.volume = self.A_single[0] * self.nelx * self.nely * self.nelz
        
        self.dof = int(K.shape[0]/self.elements_size)
        
        self.dtype = dtype


class GeneralMesh:
    def __init__(self, nodes, elements, dtype=np.float64, stiffness=lambda x0: auto_stiffness(x0, 1.0, 0.33)):
        self.nodes = nodes
        self.elements = elements
        
        self.centeroids = np.zeros((len(self.elements), self.nodes.shape[1]), dtype=dtype)
        
        self.is_uniform = True
        
        self.elements_flat = []
        self.element_sizes = np.zeros(len(self.elements), dtype=np.int32)
        self.K_flat = []
        size = len(self.elements[0])
        
        for i in range(self.elements.shape[0]):
            self.elements_flat += list(self.elements[i])
            self.element_sizes[i] = len(self.elements[i])
            self.centeroids[i] = np.mean(self.nodes[self.elements[i]], axis=0)
            if self.element_sizes[i] != size:
                self.is_uniform = False
    
        self.elements_flat = np.array(self.elements_flat, dtype=np.int32)
        self.elements_ptr = np.cumsum(self.element_sizes, dtype=np.int32)
        self.elements_ptr = np.concatenate(([0],self.elements_ptr),dtype=np.int32)

        # Clean Up Mesh To Remove Redundant Nodes
        logger.info("Checking Mesh ...")
        useful_idx = np.unique(self.elements_flat)
        if useful_idx.shape[0] != self.nodes.shape[0]:
            logger.info("Mesh has redundant nodes. Cleaning up ...")
            mapping = np.arange(self.nodes.shape[0])
            mapping = mapping[useful_idx]
            sorter = np.argsort(mapping).astype(np.int32)
            self.elements_flat = np.searchsorted(mapping, self.elements_flat, sorter=sorter).astype(np.int32)
        self.nodes = self.nodes[useful_idx]
        logger.info("Mesh Cleaned!")
        
        if self.is_uniform:
            K_temp, D_temp, B_temp, A_temp = stiffness(self.nodes[self.elements[0]])
            K_shape = K_temp.shape
            D_shape = D_temp.shape
            B_shape = B_temp.shape
            
            self.dof = int(K_shape[0]/size)
            
            self.Ks = np.zeros((len(self.elements), *K_shape), dtype=dtype)
            self.Ds = np.zeros((len(self.elements), *D_shape), dtype=dtype)
            self.Bs = np.zeros((len(self.elements), *B_shape), dtype=dtype)
            self.As = np.zeros((len(self.elements), 1), dtype=dtype)
            
            for i in range(len(self.elements)):
                self.Ks[i], self.Ds[i], self.Bs[i], self.As[i] = stiffness(self.nodes[self.elements[i]])
                
            self.elements = np.array(self.elements, dtype=np.int32)
        else:
            self.K_flat = []
            self.D_flat = []
            self.B_flat = []
            self.As = np.zeros((len(self.elements),1), dtype=dtype)
            
            self.K_ptr = np.zeros(len(self.elements)+1, dtype=np.int32)
            self.D_ptr = np.zeros(len(self.elements)+1, dtype=np.int32)
            self.B_ptr = np.zeros(len(self.elements)+1, dtype=np.int32)
            
            for i in range(len(self.elements)):
                K_temp, D_temp, B_temp, A_temp = stiffness(self.nodes[self.elements[i]])
                self.K_flat += list(K_temp.flatten())
                self.D_flat += list(D_temp.flatten())
                self.B_flat += list(B_temp.flatten())
                self.As[i] = A_temp
                
                self.K_ptr[i+1] = self.K_ptr[i] + K_temp.size
                self.D_ptr[i+1] = self.D_ptr[i] + D_temp.size
                self.B_ptr[i+1] = self.B_ptr[i] + B_temp.size

            self.K_flat = np.array(self.K_flat, dtype=dtype)
            self.D_flat = np.array(self.D_flat, dtype=dtype)
            self.B_flat = np.array(self.B_flat, dtype=dtype)
            
            self.dof = int(K_temp.shape[0]/len(self.elements[-1]))
            
        self.volume = np.sum(self.As)
        
        self.dtype = dtype


class CuGeneralMesh(GeneralMesh):
    def __init__(self, nodes, elements, dtype=np.float64, stiffness=lambda x0: auto_stiffness(x0, 1.0, 0.33)):
        super().__init__(nodes, elements, dtype, stiffness)
        
        if self.is_uniform:
            self.Ks = cp.array(self.Ks, dtype=dtype)
            self.Ds = cp.array(self.Ds, dtype=dtype)
            self.Bs = cp.array(self.Bs, dtype=dtype)
            self.As = cp.array(self.As, dtype=dtype)
        else:
            self.K_flat = cp.array(self.K_flat, dtype=dtype)
            self.D_flat = cp.array(self.D_flat, dtype=dtype)
            self.B_flat = cp.array(self.B_flat, dtype=dtype)
            self.As = cp.array(self.As, dtype=dtype)
            self.K_ptr = cp.array(self.K_ptr, dtype=cp.int32)
            self.D_ptr = cp.array(self.D_ptr, dtype=cp.int32)
            self.B_ptr = cp.array(self.B_ptr, dtype=cp.int32)
            self.elements_ptr = cp.array(self.elements_ptr, dtype=cp.int32)
            self.element_sizes = cp.array(self.element_sizes, dtype=cp.int32)
        
        self.elements_flat = cp.array(self.elements_flat, dtype=cp.int32)