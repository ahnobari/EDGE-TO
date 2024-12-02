import logging

from ..geom._mesh import StructuredMesh, GeneralMesh
from ..solvers._base import Solver
from ..MaterialModels import MaterialModel
from ..kernels._processors import StiffnessKernel
from ..geom._filters import FilterKernel
from ..core import FEA_Integrals_node_basis_parallel_flat, FEA_Integrals_node_basis_parallel_full, FEA_Integrals_node_basis_parallel
import numpy as np
from typing import Union, Optional
from scipy.spatial import KDTree
from tqdm.auto import trange

logger = logging.getLogger(__name__)


class TopOpt:
    def __init__(
        self,
        mesh: Union[StructuredMesh, GeneralMesh],
        material_model: MaterialModel,
        kernel: StiffnessKernel,
        solver: Solver,
        filter_kernel: Optional[FilterKernel] = None,
        max_iter=500,
        move=0.2,
        ch_tol=1e-4,
        fun_tol=1e-6,
        reuse_sol=True,
        solve_min_tol=np.inf,
        min_tol_patience=10,
        abandon_patience=10,
        verbose=True,
    ):
        """
        This class initializes the Topology Optimization solver object.

        Parameters:
            mesh (Union[StructuredMesh, GeneralMesh]): Mesh object for the problem.
            material_model (MaterialModel): Material model for the problem.
            kernel (StiffnessKernel): Stiffness Kernel for the problem.
            solver (Solver): Solver object for the problem.
            filter_kernel (Union[FilterKernel, None]): Filter Kernel for the problem. If not provided, it is set to None.
            max_iter (int): Maximum number of iterations for the solver. If not provided, it is set to 500.
            move (float): Move limit for the solver. If not provided, it is set to 0.2.
            ch_tol (float): Change tolerance for the solver. If not provided, it is set to 1e-4.
            fun_tol (float): Function tolerance for the solver. If not provided, it is set to 1e-6.
            reuse_sol (bool): Reuse the last solution for iterative solver. If not provided, it is set to True.
            solve_min_tol (float): Minimum tolerance for the solver to abandon. If not provided, it is set to np.inf.
            min_tol_patience (int): Patience for minimum tolerance for the solver. If not provided, it is set to 10.
            abandon_patience (int): Patience for abandoning the solver. If not provided, it is set to 10.
        """
        self.mesh = mesh
        self.material_model = material_model
        self.kernel = kernel
        self.solver = solver
        self.filter_kernel = filter_kernel
        self.max_iter = max_iter
        self.move = move
        self.ch_tol = ch_tol
        self.fun_tol = fun_tol
        self.solve_min_tol = solve_min_tol
        self.min_tol_patience = min_tol_patience
        self.abandon_patience = abandon_patience
        self.reuse_sol = reuse_sol
        self.verbose = verbose
        self.dtype = mesh.dtype

        if verbose:
            logger.setLevel(logging.INFO)

        # Set up rhs
        self.f = np.zeros([self.kernel.shape[0]], dtype=self.dtype)

        self.KDTree = None

        self.forced_material = None
        self.forced_void = None

    def FEA_integrals(self, rho):
        rho_ = self.material_model(rho, 1e6, plain=True)
        self.kernel.set_rho(rho_)
        U,residual = self.solver.solve(self.f, use_last=True)
        
        if residual > 1e-5:
            logger.warning(f"Solver residual is above 1e-5 ({residual:.4e}). Consider higher iterations (rerun this function and more iteration from prior solve will be applied).")
            
        if isinstance(self.mesh, StructuredMesh):
            strain, stress, strain_energy = FEA_Integrals_node_basis_parallel(self.mesh.K_single,
                                                                                self.mesh.D_single,
                                                                                self.mesh.B_single,
                                                                                self.kernel.elements_flat,
                                                                                rho_.shape[0],
                                                                                rho_,
                                                                                U,
                                                                                self.mesh.dof,
                                                                                self.mesh.elements_size,
                                                                                self.mesh.B_single.shape[0])
        elif self.mesh.is_uniform:
            strain, stress, strain_energy = FEA_Integrals_node_basis_parallel_full(self.mesh.Ks,
                                                                                    self.mesh.Ds,
                                                                                    self.mesh.Bs,
                                                                                    self.kernel.elements_flat,
                                                                                    rho_.shape[0],
                                                                                    rho_,
                                                                                    U,
                                                                                    self.mesh.dof,
                                                                                    self.mesh.elements.shape[1],
                                                                                    self.mesh.Bs.shape[1])
        else:
            B_size = (self.mesh.B_ptr[1]-self.mesh.B_ptr[0])//((self.mesh.elements_ptr[1]-self.mesh.elements_ptr[0])*self.mesh.dof)
            strain, stress, strain_energy = FEA_Integrals_node_basis_parallel_flat(self.mesh.K_flat,
                                                                                    self.mesh.D_flat,
                                                                                    self.mesh.B_flat,
                                                                                    self.kernel.elements_flat,
                                                                                    self.mesh.elements_ptr,
                                                                                    self.mesh.K_ptr,
                                                                                    self.mesh.B_ptr,
                                                                                    self.mesh.D_ptr,
                                                                                    rho_.shape[0],
                                                                                    rho_,
                                                                                    U,
                                                                                    self.mesh.dof,
                                                                                    B_size)
            
        if self.mesh.nodes.shape[1] == 2:
            von_mises = np.sqrt(stress[:, 0] ** 2 + stress[:, 1] ** 2 - stress[:, 0] * stress[:, 1] + 3 * stress[:, 2] ** 2)
        else:
            von_mises = np.sqrt(0.5 * ((stress[:, 0] - stress[:, 1]) ** 2 + (stress[:, 1] - stress[:, 2]) ** 2 + (stress[:, 2] - stress[:, 0]) ** 2 + 6 * (stress[:, 3] ** 2 + stress[:, 4] ** 2 + stress[:, 5] ** 2)))
    
        return strain, stress, strain_energy, von_mises
    
    def set_forced_material(self, idx):
        self.forced_material = idx

    def set_forced_void(self, idx):
        self.forced_void = idx

    def reset_forced_material(self):
        self.forced_material = None

    def reset_forced_void(self):
        self.forced_void = None

    def adjust_rho(self, rho):
        if self.forced_material is not None:
            rho[self.forced_material] = 1.0
        if self.forced_void is not None:
            rho[self.forced_void] = self.material_model.void
        return rho

    def adjust_df(self, df):
        if self.forced_material is not None:
            df[self.forced_material] = 0.0
        if self.forced_void is not None:
            df[self.forced_void] = 0.0
        return df

    def reset_BC(self):
        """
        This function resets the boundary conditions to the initial state.
        """
        self.kernel.set_constraints([])
        self.kernel.has_cons = False

    def reset_F(self):
        """
        This function resets the forces to the initial state.
        """
        self.f[:] = 0

    def add_BCs(self, positions, BCs):
        """
        This function adds boundary conditions to the solver.

        Parameters:
            positions (np.array): Array with the nodal positions where the BCs are to be applied.
            BCs (np.array): Array with the BCs to be applied 0 on dimensions not to be constrained and 1 for constrained directions. Shape (n_nodes, dim)
        """
        if self.KDTree is None:
            self.KDTree = KDTree(self.mesh.nodes)
        _, idx = self.KDTree.query(positions)

        if BCs.shape[0] != idx.shape[0] and BCs.shape[0] > 1:
            raise ValueError(
                "BCs shape does not match positions shape and is not broadcastable."
            )

        if BCs.shape[0] == 1:

            for i in range(self.mesh.dof):
                if BCs[0, i]:
                    cons = idx * self.mesh.dof + i
                    self.kernel.add_constraints(cons)
        else:
            for i in range(self.mesh.dof):
                cons = idx[BCs[:, i]] * self.mesh.dof + i
                self.kernel.add_constraints(cons)

    def add_Forces(self, positions, Fs):
        """
        This function adds forces to the solver.

        Parameters:
            positions (np.array): Array with the nodal positions where the forces are to be applied.
            Fs (np.array): Array with the forces to be applied. Shape (n_nodes, dim)
        """
        if self.KDTree is None:
            self.KDTree = KDTree(self.mesh.nodes)

        _, idx = self.KDTree.query(positions)

        if Fs.shape[0] != idx.shape[0] and Fs.shape[0] > 1:
            raise ValueError(
                "Fs shape does not match positions shape and is not broadcastable."
            )

        Fs = np.array(Fs, dtype=self.dtype)
        if Fs.shape[0] == 1:
            for i in range(self.mesh.dof):
                self.f[idx * self.mesh.dof + i] = Fs[0, i]
        else:
            for i in range(self.mesh.dof):
                self.f[idx * self.mesh.dof + i] = Fs[:, i]

    def add_BC_nodal(self, node_ids, BCs):
        """
        This function adds boundary conditions to the solver.

        Parameters:
            node_ids (np.array): Array with the nodal indecies where the BCs are to be applied.
            BCs (np.array): Array with the BCs to be applied 0 on dimensions not to be constrained and 1 for constrained directions. Shape (n_nodes, dim)
        """
        if BCs.shape[0] != node_ids.shape[0] and BCs.shape[0] > 1:
            raise ValueError(
                "BCs shape does not match positions shape and is not broadcastable."
            )

        if BCs.shape[0] == 1:
            for i in range(self.mesh.dof):
                if BCs[0, i]:
                    cons = node_ids * self.mesh.dof + i
                    self.kernel.add_constraints(cons)
        else:
            for i in range(self.mesh.dof):
                cons = node_ids[BCs[:, i]] * self.mesh.dof + i
                self.kernel.add_constraints(cons)

    def add_F_nodal(self, node_ids, Fs):
        """
        This function adds forces to the solver.

        Parameters:
            node_ids (np.array): Array with the nodal indecies where the forces are to be applied.
            Fs (np.array): Array with the forces to be applied. Shape (n_nodes, dim)
        """
        if Fs.shape[0] != node_ids.shape[0] and Fs.shape[0] > 1:
            raise ValueError(
                "Fs shape does not match positions shape and is not broadcastable."
            )

        Fs = np.array(Fs, dtype=self.dtype)

        if Fs.shape[0] == 1:
            for i in range(self.mesh.dof):
                self.f[node_ids * self.mesh.dof + i] = Fs[0, i]
        else:
            for i in range(self.mesh.dof):
                self.f[node_ids[:, 0] * self.mesh.dof + i] = Fs[:, i]

    def update_design(self, rho, df):
        """
        This function updates the design based on the SIMP method.

        Parameters:
            rho (np.array): Density array.
            df (np.array): Sensitivity array.
        Returns:
            rho_new (np.array): Updated density array.
        """
        rho = rho.reshape(-1, 1)
        xU = np.clip(rho + self.move, 0.0, 1.0)
        xL = np.clip(rho - self.move, 0.0, 1.0)

        ocP = self.material_model.ocP(df, self.mesh.As, rho, np=np)
        ocP = np.maximum(1e-12, ocP)

        size = self.material_model.base_properties()["n_sets"]
        l1 = 1e-9 * np.ones(size, dtype=self.dtype)
        l2 = 1e9 * np.ones(size, dtype=self.dtype)

        lmid = 0.5 * (l1 + l2)

        rho_new = np.maximum(
            0, np.maximum(xL, np.minimum(1.0, np.minimum(xU, ocP / lmid)))
        )

        while np.any((l2 - l1) / (l2 + l1) > 1e-4):
            lmid = 0.5 * (l1 + l2)
            rho_new = np.maximum(
                0, np.maximum(xL, np.minimum(1.0, np.minimum(xU, ocP / lmid)))
            )

            valids = self.material_model.evaluate_constraint(
                rho_new, self.mesh.As, self.mesh.volume, np=np
            )

            l2[valids] = lmid[valids]
            l1[~valids] = lmid[~valids]

        return rho_new.reshape(-1)

    def solve_and_process(self, rho, iteration):

        rho_ = self.filter_kernel.dot(rho)
        # rho_ = self.adjust_rho(rho_)
        rho__ = self.material_model(rho_, iteration, np=np)
        
        self.kernel.set_rho(rho__)
        
        U, residual = self.solver.solve(self.f, use_last=self.reuse_sol)

        compliance = self.f.dot(U)

        df = self.kernel.process_grad(U)

        dr = self.material_model.grad(rho_, iteration, np=np) * df

        dr = dr.reshape(dr.shape[0], -1)

        for i in range(dr.shape[1]):
            dr[:, i] = self.filter_kernel._rmatvec(dr[:, i])

        return compliance, dr, residual

    def optimize(
        self,
        save_comp_history=False,
        save_change_history=False,
        save_rho_history=False,
        rho_init=None,
    ):
        """
        This function performs the topology optimization.

        Returns:
            rho (np.array): Optimized Density array.
            flag (bool): Flag indicating if the optimization converged.
        """

        flag = False
        change_f = np.inf
        change = np.inf

        if rho_init is None:
            rho = self.material_model.init_desvars(len(self.mesh.elements), dtype=self.dtype)
        else:
            rho = self.material_model(rho, 0, plain=True)

        self.kernel.set_rho(rho)

        # Initial Solve
        comp, df, res = self.solve_and_process(rho, 0)

        prog = trange(self.max_iter, disable=logger.getEffectiveLevel() > logging.INFO)

        if save_comp_history:
            comp_history = []

        if save_change_history:
            change_history = []

        if save_rho_history:
            rho_history = []

        for i in prog:
            
            rho_old = np.copy(rho)
            comp_old = np.copy(comp)

            rho = self.update_design(rho, df)

            comp, df, res = self.solve_and_process(rho, i)

            change = np.sqrt(np.square(rho - rho_old).mean())
            change_f = np.abs(comp - comp_old) / comp

            if save_change_history:
                change_history.append(change)

            if save_rho_history:
                rho_history.append(rho)

            if save_comp_history:
                comp_history.append(comp)

            prog.set_postfix_str(
                f"Compliance: {comp:.4e}, Change: {change:.4e}, Function Change: {change_f:.4e}, Residual: {res:.4e}"
            )

            if (
                change < self.ch_tol
                and change_f < self.fun_tol
                and self.material_model.is_terminal(i)
            ):
                flag = True
                break

        hist = {}
        if save_comp_history:
            hist["comp_history"] = comp_history
        if save_change_history:
            hist["change_history"] = change_history
        if save_rho_history:
            hist["rho_history"] = rho_history

        return rho, flag, hist
