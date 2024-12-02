from ._2d import plot_mesh_2D, plot_problem_2D, plot_field_2D
from ._3d import plot_problem_3D, plot_mesh_3D, plot_field_3D
from ..TopOpt import TopOpt, CuTopOpt
from typing import Union

class Plotter:
    def __init__(self, problem: Union[TopOpt, CuTopOpt]):
        self.problem = problem
        self.nd = problem.mesh.nodes.shape[1]
        
    def _plot_problem(self, ax = None, **kwargs):
        if self.nd == 2:
            return plot_problem_2D(self.problem, ax=ax, **kwargs)
        elif self.nd == 3:
            return plot_problem_3D(self.problem, **kwargs)

    def _plot_mesh(self, ax=None, **kwargs):
        if self.nd == 2:
            return plot_mesh_2D(self.problem.mesh, ax=ax, **kwargs)
        elif self.nd == 3:
            return plot_mesh_3D(self.problem.mesh, **kwargs)

    def display_mesh(self, **kwargs):
        return self._plot_mesh(**kwargs)
    
    def display(self, **kwargs):
        return self._plot_problem(**kwargs)
    
    def display_solution(self, rho, **kwargs):
        return self._plot_problem(rho=rho, **kwargs)
    
    def display_solution_mesh(self, rho, **kwargs):
        return self._plot_mesh(rho=rho, **kwargs)
    
    def display_field(self, field, **kwargs):
        if self.nd == 2:
            return plot_field_2D(self.problem.mesh, field, **kwargs)
        else:
            return plot_field_3D(self.problem.mesh, field, **kwargs)