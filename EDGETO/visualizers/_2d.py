import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import markers
from matplotlib.path import Path
from ..geom import StructuredMesh2D, GeneralMesh, CuStructuredMesh2D, CuGeneralMesh
from typing import Union, Tuple, List
from ..TopOpt import TopOpt, CuTopOpt

def align_marker(marker, halign="center", valign="middle"):

    if isinstance(halign, (str)):
        halign = {
            "right": -1.0,
            "middle": 0.0,
            "center": 0.0,
            "left": 1.0,
        }[halign]

    if isinstance(valign, (str)):
        valign = {
            "top": -1.0,
            "middle": 0.0,
            "center": 0.0,
            "bottom": 1.0,
        }[valign]

    bm = markers.MarkerStyle(marker)

    m_arr = bm.get_path().transformed(bm.get_transform()).vertices

    m_arr[:, 0] += halign / 2
    m_arr[:, 1] += valign / 2

    return Path(m_arr, bm.get_path().codes)


def plot_problem_2D(
    problem: Union[TopOpt, CuTopOpt],
    ax=None,
    face_color="grey",
    edge_color="black",
    x_color="tomato",
    y_color="royalblue",
    f_color="#8e0000",
    rho=None,
    **kwargs,
):
    if isinstance(problem.mesh, CuStructuredMesh2D):
        nodes = problem.mesh.nodes.get()
        elements = problem.mesh.elements.get()
    else:
        nodes = problem.mesh.nodes
        elements = problem.mesh.elements
        
    if isinstance(problem, CuTopOpt):
        c = problem.kernel.constraints.get().reshape(-1, problem.mesh.dof)
        f = problem.f.get().reshape(-1, problem.mesh.dof)
    else:
        c = problem.kernel.constraints.reshape(-1, problem.mesh.dof)
        f = problem.f.reshape(-1, problem.mesh.dof)

    if rho is not None:
        elements = elements[rho>0.5]
    
    if ax is None:
        ax = plt.gca()

    y = nodes[:, 0]
    z = nodes[:, 1]

    def quatplot(y, z, quatrangles, ax=None, **kwargs):

        if not ax:
            ax = plt.gca()
        yz = np.c_[y, z]
        verts = yz[quatrangles]
        pc = matplotlib.collections.PolyCollection(verts, **kwargs)
        ax.add_collection(pc)
        ax.autoscale()

    ax.set_aspect("equal")

    elements_ = []
    for e in elements:
        if len(e) == 4:
            elements_.append([e[0], e[1], e[2], e[3]])
        else:
            elements_.append([e[0], e[1], e[2], e[2]])

    quatplot(y, z, np.asarray(elements_), ax=ax, color=edge_color, facecolor=face_color)

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")

    x_bc = c[:, 0] != 0
    y_bc = c[:, 1] != 0

    ax.scatter(
        nodes[x_bc][:, 0],
        nodes[x_bc][:, 1],
        marker=align_marker(">", "right", "middle"),
        s=500,
        color=x_color,
        alpha=0.7,
    )
    ax.scatter(
        nodes[y_bc][:, 0],
        nodes[y_bc][:, 1],
        marker=align_marker("^", "middle", "top"),
        s=500,
        color=y_color,
        alpha=0.7,
    )

    force_nodes = (f != 0).sum(1) > 0

    ax.quiver(
        nodes[force_nodes, 0],
        nodes[force_nodes, 1],
        f[force_nodes, 0] / np.abs(f).max(),
        f[force_nodes, 1] / np.abs(f).max(),
        color=f_color,
        scale=15,
        width=0.005,
    )

    return ax


def plot_mesh_2D(
    mesh: Union[StructuredMesh2D, GeneralMesh, CuStructuredMesh2D, CuGeneralMesh],
    ax=None,
    face_color="grey", 
    edge_color="black",
    rho=None,
    **kwargs
):
    if isinstance(mesh, GeneralMesh) or isinstance(mesh, CuGeneralMesh):
        if mesh.nodes.shape[1] != 2:
            raise ValueError("This function only supports 2D meshes")

    if ax is None:
        ax = plt.gca()

    y = mesh.nodes[:, 0]
    z = mesh.nodes[:, 1]

    if isinstance(mesh, CuStructuredMesh2D):
        y = y.get()
        z = z.get()
        
    def quatplot(y, z, quatrangles, ax=None, **kwargs):

        if not ax:
            ax = plt.gca()
        yz = np.c_[y, z]
        verts = yz[quatrangles]
        pc = matplotlib.collections.PolyCollection(verts, **kwargs)
        ax.add_collection(pc)
        ax.autoscale()

    ax.set_aspect("equal")

    elements_ = []
    if isinstance(mesh, CuStructuredMesh2D):
        elements = mesh.elements.get()
    else:
        elements = mesh.elements
        
    if rho is not None:
        elements = elements[rho>0.5]
        
    for e in elements:
        if len(e) == 4:
            elements_.append([e[0], e[1], e[2], e[3]])
        else:
            elements_.append([e[0], e[1], e[2], e[2]])

    quatplot(y, z, np.asarray(elements_), ax=ax, color=edge_color, facecolor=face_color)

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")

    return ax

def plot_field_2D(
    mesh: Union[StructuredMesh2D, GeneralMesh, CuStructuredMesh2D, CuGeneralMesh],
    field,
    rho = None,
    ax=None,
    face_color="grey",
    edge_color="black",
    colormap='viridis',
    show_colorbar=True,
    colorbar_label=None,
    **kwargs,
):
    if isinstance(mesh, CuStructuredMesh2D):
        nodes = mesh.nodes.get()
        elements = mesh.elements.get()
    else:
        nodes = mesh.nodes
        elements = mesh.elements

    if rho is not None:
        elements = elements[rho>0.5]
        field = field[rho>0.5]
    
    if ax is None:
        ax = plt.gca()

    y = nodes[:, 0]
    z = nodes[:, 1]

    def quatplot(y, z, quatrangles, field_values, ax=None, **kwargs):
        if not ax:
            ax = plt.gca()
        yz = np.c_[y, z]
        verts = yz[quatrangles]
        pc = matplotlib.collections.PolyCollection(verts, **kwargs)
        pc.set_array(field_values)
        ax.add_collection(pc)
        ax.autoscale()
        return pc

    ax.set_aspect("equal")

    elements_ = []
    for e in elements:
        if len(e) == 4:
            elements_.append([e[0], e[1], e[2], e[3]])
        else:
            elements_.append([e[0], e[1], e[2], e[2]])

    pc = quatplot(y, z, np.asarray(elements_), field, ax=ax, 
                 edgecolor=edge_color, cmap=colormap, **kwargs)

    if show_colorbar:
        cbar = plt.colorbar(pc, ax=ax)
        if colorbar_label:
            cbar.set_label(colorbar_label)

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")

    return ax