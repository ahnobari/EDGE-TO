from ._cholmod import CHOLMOD
from ._iteratives import BiCGSTAB, CG, GMRES
from ._splu import SPLU, SPSOLVE
from ._multigrid import MultiGrid
from ._gpu import CG as CuCG, GMRES as CuGMRES, SPSOLVE as CuSPSOLVE
from ._multigrid_cuda import MultiGrid as CuMultiGrid