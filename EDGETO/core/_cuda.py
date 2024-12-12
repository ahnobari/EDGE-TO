import os

import cupy as cp
from functools import lru_cache

# Load the kernel codes from the .cu files

with open(os.path.join(os.path.dirname(__file__), "_templated_cuda.cu"), "r") as f:
    main_code = f.read()

with open(os.path.join(os.path.dirname(__file__), "_mgm_cuda.cu"), "r") as f:
    mgm_code = f.read()

named_functions = [
    "pop_constraints_vec_cuda_kernel<double>",
    "pop_constraints_vec_cuda_kernel<float>",
    "recover_constraints_vec_cuda_kernel<double>",
    "recover_constraints_vec_cuda_kernel<float>",
    "pop_constraints_cuda_kernel<double>",
    "pop_constraints_cuda_kernel<float>",
    "recover_constraints_cuda_kernel<double>",
    "recover_constraints_cuda_kernel<float>",
    "mat_vec_node_basis_parallel_flat_cuda_kernel<double>",
    "mat_vec_node_basis_parallel_flat_cuda_kernel<float>",
    "mat_vec_node_basis_parallel_full_cuda_kernel<double>",
    "mat_vec_node_basis_parallel_full_cuda_kernel<float>",
    "mat_vec_node_basis_parallel_cuda_kernel<double>",
    "mat_vec_node_basis_parallel_cuda_kernel<float>",
    "process_dk_kernel_cuda<double>",
    "process_dk_kernel_cuda<float>",
    "process_dk_full_kernel_cuda<double>",
    "process_dk_full_kernel_cuda<float>",
    "process_dk_flat_kernel_cuda<double>",
    "process_dk_flat_kernel_cuda<float>",
    "restriction_3d_kernel<double>",
    "restriction_3d_kernel<float>",
    "restriction_2d_kernel<double>",
    "restriction_2d_kernel<float>",
    "prolongation_3d_kernel<double>",
    "prolongation_3d_kernel<float>",
    "prolongation_2d_kernel<double>",
    "prolongation_2d_kernel<float>",
    "csrcsr_matmat_kernel<double>",
    "csrcsr_matmat_kernel<float>",
    "matmat_node_basis_parallel_kernel<double>",
    "matmat_node_basis_parallel_kernel<float>",
    "matmat_node_basis_full_parallel_kernel<double>",
    "matmat_node_basis_full_parallel_kernel<float>",
    "matmat_node_basis_flat_parallel_kernel<double>",
    "matmat_node_basis_flat_parallel_kernel<float>",
    "mat_vec_node_basis_parallel_flat_wcon_cuda_kernel<double>",
    "mat_vec_node_basis_parallel_flat_wcon_cuda_kernel<float>",
    "mat_vec_node_basis_parallel_full_wcon_cuda_kernel<double>",
    "mat_vec_node_basis_parallel_full_wcon_cuda_kernel<float>",
    "mat_vec_node_basis_parallel_wcon_cuda_kernel<double>",
    "mat_vec_node_basis_parallel_wcon_cuda_kernel<float>",
    "matmat_node_basis_full_parallel_wcon_kernel<double>",
    "matmat_node_basis_full_parallel_wcon_kernel<float>",
    "matmat_node_basis_flat_parallel_wcon_kernel<double>",
    "matmat_node_basis_flat_parallel_wcon_kernel<float>",
    "matmat_node_basis_parallel_wcon_kernel<double>",
    "matmat_node_basis_parallel_wcon_kernel<float>",
    "get_diagonal_node_basis_cuda_kernel<double>",
    "get_diagonal_node_basis_cuda_kernel<float>",
    "get_diagonal_node_basis_full_cuda_kernel<double>",
    "get_diagonal_node_basis_full_cuda_kernel<float>",
    "get_diagonal_node_basis_flat_cuda_kernel<double>",
    "get_diagonal_node_basis_flat_cuda_kernel<float>",
    "restriction_matrix_2d_kernel<double>",
    "restriction_matrix_2d_kernel<float>",
    "restriction_matrix_3d_kernel<double>",
    "restriction_matrix_3d_kernel<float>",
    "prolongation_matrix_2d_kernel<double>",
    "prolongation_matrix_2d_kernel<float>",
    "prolongation_matrix_3d_kernel<double>",
    "prolongation_matrix_3d_kernel<float>",
    "generate_nodes_2d<double>",
    "generate_nodes_2d<float>",
    "generate_nodes_3d<double>",
    "generate_nodes_3d<float>",
    "apply_filter_2D_kernel<double>",
    "apply_filter_2D_kernel<float>",
    "apply_filter_3D_kernel<double>",
    "apply_filter_3D_kernel<float>",
    "get_filter_2D_weights_kernel<double>",
    "get_filter_2D_weights_kernel<float>",
    "get_filter_3D_weights_kernel<double>",
    "get_filter_3D_weights_kernel<float>",
    "apply_filter_2D_transpose_kernel<double>",
    "apply_filter_2D_transpose_kernel<float>",
    "apply_filter_3D_transpose_kernel<double>",
    "apply_filter_3D_transpose_kernel<float>",
    "FEA_Integrals_node_basis_parallel_cuda_kernel<double>",
    "FEA_Integrals_node_basis_parallel_cuda_kernel<float>",
    "FEA_Integrals_node_basis_parallel_full_cuda_kernel<double>",
    "FEA_Integrals_node_basis_parallel_full_cuda_kernel<float>",
    "FEA_Integrals_node_basis_parallel_flat_cuda_kernel<double>",
    "FEA_Integrals_node_basis_parallel_flat_cuda_kernel<float>"
    ]

_cuda_module = cp.RawModule(code=main_code, options=('-std=c++11',), name_expressions=named_functions)

_cuda_module_mgm = cp.RawModule(code=mgm_code, options=('-std=c++11',), name_expressions=[
    'get_restricted_3d_l0_nnz_based<double>',
    'get_restricted_3d_l0_nnz_based<float>',
    'get_restricted_3d_l1p_nnz_based<double>',
    'get_restricted_3d_l1p_nnz_based<float>',
    'get_restricted_2d_l0_nnz_based<double>',
    'get_restricted_2d_l0_nnz_based<float>',
    'get_restricted_2d_l1p_nnz_based<double>',
    'get_restricted_2d_l1p_nnz_based<float>',
    'SOR_l0<double>',
    'SOR_l0<float>',
    'SOR_l1p<double>',
    'SOR_l1p<float>',
    'csr_matvec<double>',
    'csr_matvec<float>',
    'csr_diagonal<double>',
    'csr_diagonal<float>',
])

csr_diagonal_double = _cuda_module_mgm.get_function("csr_diagonal<double>")
csr_diagonal_float = _cuda_module_mgm.get_function("csr_diagonal<float>")

def csr_diagonal(*args):
    if args[-1][0].dtype == cp.float64:
        return csr_diagonal_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return csr_diagonal_float(*args)
    else:
        raise ValueError("Unsupported dtype")

csr_matvec_double = _cuda_module_mgm.get_function("csr_matvec<double>")
csr_matvec_float = _cuda_module_mgm.get_function("csr_matvec<float>")

def csr_matvec(*args):
    if args[-1][0].dtype == cp.float64:
        return csr_matvec_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return csr_matvec_float(*args)
    else:
        raise ValueError("Unsupported dtype")

SOR_l0_double = _cuda_module_mgm.get_function("SOR_l0<double>")
SOR_l0_float = _cuda_module_mgm.get_function("SOR_l0<float>")

def SOR_l0(*args):
    if args[-1][0].dtype == cp.float64:
        return SOR_l0_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return SOR_l0_float(*args)
    else:
        raise ValueError("Unsupported dtype")

SOR_l1p_double = _cuda_module_mgm.get_function("SOR_l1p<double>")
SOR_l1p_float = _cuda_module_mgm.get_function("SOR_l1p<float>")

def SOR_l1p(*args):
    if args[-1][0].dtype == cp.float64:
        return SOR_l1p_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return SOR_l1p_float(*args)
    else:
        raise ValueError("Unsupported dtype")

get_restricted_2d_l0_nnz_based_double = _cuda_module_mgm.get_function("get_restricted_2d_l0_nnz_based<double>")
get_restricted_2d_l0_nnz_based_float = _cuda_module_mgm.get_function("get_restricted_2d_l0_nnz_based<float>")

def get_restricted_2d_l0_nnz_based(*args):
    if args[-1][0].dtype == cp.float64:
        return get_restricted_2d_l0_nnz_based_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return get_restricted_2d_l0_nnz_based_float(*args)
    else:
        raise ValueError("Unsupported dtype")
    
get_restricted_2d_l1p_nnz_based_double = _cuda_module_mgm.get_function("get_restricted_2d_l1p_nnz_based<double>")
get_restricted_2d_l1p_nnz_based_float = _cuda_module_mgm.get_function("get_restricted_2d_l1p_nnz_based<float>")

def get_restricted_2d_l1p_nnz_based(*args):
    if args[-1][0].dtype == cp.float64:
        return get_restricted_2d_l1p_nnz_based_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return get_restricted_2d_l1p_nnz_based_float(*args)
    else:
        raise ValueError("Unsupported dtype")

get_restricted_3d_l1p_nnz_based_double = _cuda_module_mgm.get_function("get_restricted_3d_l1p_nnz_based<double>")
get_restricted_3d_l1p_nnz_based_float = _cuda_module_mgm.get_function("get_restricted_3d_l1p_nnz_based<float>")

def get_restricted_3d_l1p_nnz_based(*args):
    if args[-1][0].dtype == cp.float64:
        return get_restricted_3d_l1p_nnz_based_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return get_restricted_3d_l1p_nnz_based_float(*args)
    else:
        raise ValueError("Unsupported dtype")

get_restricted_3d_l0_nnz_based_double = _cuda_module_mgm.get_function("get_restricted_3d_l0_nnz_based<double>")
get_restricted_3d_l0_nnz_based_float = _cuda_module_mgm.get_function("get_restricted_3d_l0_nnz_based<float>")

def get_restricted_3d_l0_nnz_based(*args):
    if args[-1][0].dtype == cp.float64:
        return get_restricted_3d_l0_nnz_based_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return get_restricted_3d_l0_nnz_based_float(*args)
    else:
        raise ValueError("Unsupported dtype")

pop_constraints_vec_cuda_kernel_double = _cuda_module.get_function("pop_constraints_vec_cuda_kernel<double>")
pop_constraints_vec_cuda_kernel_float = _cuda_module.get_function("pop_constraints_vec_cuda_kernel<float>")

def pop_constraints_vec_cuda_kernel(*args):
    if args[-1][2].dtype == cp.float64:
        return pop_constraints_vec_cuda_kernel_double(*args)
    elif args[-1][2].dtype == cp.float32:
        return pop_constraints_vec_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



recover_constraints_vec_cuda_kernel_double = _cuda_module.get_function("recover_constraints_vec_cuda_kernel<double>")
recover_constraints_vec_cuda_kernel_float = _cuda_module.get_function("recover_constraints_vec_cuda_kernel<float>")

def recover_constraints_vec_cuda_kernel(*args):
    if args[-1][2].dtype == cp.float64:
        return recover_constraints_vec_cuda_kernel_double(*args)
    elif args[-1][2].dtype == cp.float32:
        return recover_constraints_vec_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



pop_constraints_cuda_kernel_double = _cuda_module.get_function("pop_constraints_cuda_kernel<double>")
pop_constraints_cuda_kernel_float = _cuda_module.get_function("pop_constraints_cuda_kernel<float>")

def pop_constraints_cuda_kernel(*args):
    if args[-1][2].dtype == cp.float64:
        return pop_constraints_cuda_kernel_double(*args)
    elif args[-1][2].dtype == cp.float32:
        return pop_constraints_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



recover_constraints_cuda_kernel_double = _cuda_module.get_function("recover_constraints_cuda_kernel<double>")
recover_constraints_cuda_kernel_float = _cuda_module.get_function("recover_constraints_cuda_kernel<float>")

def recover_constraints_cuda_kernel(*args):
    if args[-1][2].dtype == cp.float64:
        return recover_constraints_cuda_kernel_double(*args)
    elif args[-1][2].dtype == cp.float32:
        return recover_constraints_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



mat_vec_node_basis_parallel_flat_cuda_kernel_double = _cuda_module.get_function("mat_vec_node_basis_parallel_flat_cuda_kernel<double>")
mat_vec_node_basis_parallel_flat_cuda_kernel_float = _cuda_module.get_function("mat_vec_node_basis_parallel_flat_cuda_kernel<float>")

def mat_vec_node_basis_parallel_flat_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return mat_vec_node_basis_parallel_flat_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return mat_vec_node_basis_parallel_flat_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



mat_vec_node_basis_parallel_full_cuda_kernel_double = _cuda_module.get_function("mat_vec_node_basis_parallel_full_cuda_kernel<double>")
mat_vec_node_basis_parallel_full_cuda_kernel_float = _cuda_module.get_function("mat_vec_node_basis_parallel_full_cuda_kernel<float>")

def mat_vec_node_basis_parallel_full_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return mat_vec_node_basis_parallel_full_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return mat_vec_node_basis_parallel_full_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



mat_vec_node_basis_parallel_cuda_kernel_double = _cuda_module.get_function("mat_vec_node_basis_parallel_cuda_kernel<double>")
mat_vec_node_basis_parallel_cuda_kernel_float = _cuda_module.get_function("mat_vec_node_basis_parallel_cuda_kernel<float>")

def mat_vec_node_basis_parallel_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return mat_vec_node_basis_parallel_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return mat_vec_node_basis_parallel_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



process_dk_kernel_cuda_double = _cuda_module.get_function("process_dk_kernel_cuda<double>")
process_dk_kernel_cuda_float = _cuda_module.get_function("process_dk_kernel_cuda<float>")

def process_dk_kernel_cuda(*args):
    if args[-1][0].dtype == cp.float64:
        return process_dk_kernel_cuda_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return process_dk_kernel_cuda_float(*args)
    else:
        raise ValueError("Unsupported dtype")



process_dk_full_kernel_cuda_double = _cuda_module.get_function("process_dk_full_kernel_cuda<double>")
process_dk_full_kernel_cuda_float = _cuda_module.get_function("process_dk_full_kernel_cuda<float>")

def process_dk_full_kernel_cuda(*args):
    if args[-1][0].dtype == cp.float64:
        return process_dk_full_kernel_cuda_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return process_dk_full_kernel_cuda_float(*args)
    else:
        raise ValueError("Unsupported dtype")



process_dk_flat_kernel_cuda_double = _cuda_module.get_function("process_dk_flat_kernel_cuda<double>")
process_dk_flat_kernel_cuda_float = _cuda_module.get_function("process_dk_flat_kernel_cuda<float>")

def process_dk_flat_kernel_cuda(*args):
    if args[-1][0].dtype == cp.float64:
        return process_dk_flat_kernel_cuda_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return process_dk_flat_kernel_cuda_float(*args)
    else:
        raise ValueError("Unsupported dtype")



restriction_3d_kernel_double = _cuda_module.get_function("restriction_3d_kernel<double>")
restriction_3d_kernel_float = _cuda_module.get_function("restriction_3d_kernel<float>")

def restriction_3d_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return restriction_3d_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return restriction_3d_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



restriction_2d_kernel_double = _cuda_module.get_function("restriction_2d_kernel<double>")
restriction_2d_kernel_float = _cuda_module.get_function("restriction_2d_kernel<float>")

def restriction_2d_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return restriction_2d_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return restriction_2d_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



prolongation_3d_kernel_double = _cuda_module.get_function("prolongation_3d_kernel<double>")
prolongation_3d_kernel_float = _cuda_module.get_function("prolongation_3d_kernel<float>")

def prolongation_3d_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return prolongation_3d_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return prolongation_3d_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



prolongation_2d_kernel_double = _cuda_module.get_function("prolongation_2d_kernel<double>")
prolongation_2d_kernel_float = _cuda_module.get_function("prolongation_2d_kernel<float>")

def prolongation_2d_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return prolongation_2d_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return prolongation_2d_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



csrcsr_matmat_kernel_double = _cuda_module.get_function("csrcsr_matmat_kernel<double>")
csrcsr_matmat_kernel_float = _cuda_module.get_function("csrcsr_matmat_kernel<float>")

def csrcsr_matmat_kernel(*args):
    if args[-1][2].dtype == cp.float64:
        return csrcsr_matmat_kernel_double(*args)
    elif args[-1][2].dtype == cp.float32:
        return csrcsr_matmat_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



matmat_node_basis_parallel_kernel_double = _cuda_module.get_function("matmat_node_basis_parallel_kernel<double>")
matmat_node_basis_parallel_kernel_float = _cuda_module.get_function("matmat_node_basis_parallel_kernel<float>")

def matmat_node_basis_parallel_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return matmat_node_basis_parallel_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return matmat_node_basis_parallel_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



matmat_node_basis_full_parallel_kernel_double = _cuda_module.get_function("matmat_node_basis_full_parallel_kernel<double>")
matmat_node_basis_full_parallel_kernel_float = _cuda_module.get_function("matmat_node_basis_full_parallel_kernel<float>")

def matmat_node_basis_full_parallel_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return matmat_node_basis_full_parallel_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return matmat_node_basis_full_parallel_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



matmat_node_basis_flat_parallel_kernel_double = _cuda_module.get_function("matmat_node_basis_flat_parallel_kernel<double>")
matmat_node_basis_flat_parallel_kernel_float = _cuda_module.get_function("matmat_node_basis_flat_parallel_kernel<float>")

def matmat_node_basis_flat_parallel_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return matmat_node_basis_flat_parallel_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return matmat_node_basis_flat_parallel_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



mat_vec_node_basis_parallel_flat_wcon_cuda_kernel_double = _cuda_module.get_function("mat_vec_node_basis_parallel_flat_wcon_cuda_kernel<double>")
mat_vec_node_basis_parallel_flat_wcon_cuda_kernel_float = _cuda_module.get_function("mat_vec_node_basis_parallel_flat_wcon_cuda_kernel<float>")

def mat_vec_node_basis_parallel_flat_wcon_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return mat_vec_node_basis_parallel_flat_wcon_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return mat_vec_node_basis_parallel_flat_wcon_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



mat_vec_node_basis_parallel_full_wcon_cuda_kernel_double = _cuda_module.get_function("mat_vec_node_basis_parallel_full_wcon_cuda_kernel<double>")
mat_vec_node_basis_parallel_full_wcon_cuda_kernel_float = _cuda_module.get_function("mat_vec_node_basis_parallel_full_wcon_cuda_kernel<float>")

def mat_vec_node_basis_parallel_full_wcon_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return mat_vec_node_basis_parallel_full_wcon_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return mat_vec_node_basis_parallel_full_wcon_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



mat_vec_node_basis_parallel_wcon_cuda_kernel_double = _cuda_module.get_function("mat_vec_node_basis_parallel_wcon_cuda_kernel<double>")
mat_vec_node_basis_parallel_wcon_cuda_kernel_float = _cuda_module.get_function("mat_vec_node_basis_parallel_wcon_cuda_kernel<float>")

def mat_vec_node_basis_parallel_wcon_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return mat_vec_node_basis_parallel_wcon_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return mat_vec_node_basis_parallel_wcon_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



matmat_node_basis_full_parallel_wcon_kernel_double = _cuda_module.get_function("matmat_node_basis_full_parallel_wcon_kernel<double>")
matmat_node_basis_full_parallel_wcon_kernel_float = _cuda_module.get_function("matmat_node_basis_full_parallel_wcon_kernel<float>")

def matmat_node_basis_full_parallel_wcon_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return matmat_node_basis_full_parallel_wcon_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return matmat_node_basis_full_parallel_wcon_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



matmat_node_basis_flat_parallel_wcon_kernel_double = _cuda_module.get_function("matmat_node_basis_flat_parallel_wcon_kernel<double>")
matmat_node_basis_flat_parallel_wcon_kernel_float = _cuda_module.get_function("matmat_node_basis_flat_parallel_wcon_kernel<float>")

def matmat_node_basis_flat_parallel_wcon_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return matmat_node_basis_flat_parallel_wcon_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return matmat_node_basis_flat_parallel_wcon_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



matmat_node_basis_parallel_wcon_kernel_double = _cuda_module.get_function("matmat_node_basis_parallel_wcon_kernel<double>")
matmat_node_basis_parallel_wcon_kernel_float = _cuda_module.get_function("matmat_node_basis_parallel_wcon_kernel<float>")

def matmat_node_basis_parallel_wcon_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return matmat_node_basis_parallel_wcon_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return matmat_node_basis_parallel_wcon_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



get_diagonal_node_basis_cuda_kernel_double = _cuda_module.get_function("get_diagonal_node_basis_cuda_kernel<double>")
get_diagonal_node_basis_cuda_kernel_float = _cuda_module.get_function("get_diagonal_node_basis_cuda_kernel<float>")

def get_diagonal_node_basis_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return get_diagonal_node_basis_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return get_diagonal_node_basis_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



get_diagonal_node_basis_full_cuda_kernel_double = _cuda_module.get_function("get_diagonal_node_basis_full_cuda_kernel<double>")
get_diagonal_node_basis_full_cuda_kernel_float = _cuda_module.get_function("get_diagonal_node_basis_full_cuda_kernel<float>")

def get_diagonal_node_basis_full_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return get_diagonal_node_basis_full_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return get_diagonal_node_basis_full_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



get_diagonal_node_basis_flat_cuda_kernel_double = _cuda_module.get_function("get_diagonal_node_basis_flat_cuda_kernel<double>")
get_diagonal_node_basis_flat_cuda_kernel_float = _cuda_module.get_function("get_diagonal_node_basis_flat_cuda_kernel<float>")

def get_diagonal_node_basis_flat_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return get_diagonal_node_basis_flat_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return get_diagonal_node_basis_flat_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



restriction_matrix_2d_kernel_double = _cuda_module.get_function("restriction_matrix_2d_kernel<double>")
restriction_matrix_2d_kernel_float = _cuda_module.get_function("restriction_matrix_2d_kernel<float>")

def restriction_matrix_2d_kernel(*args):
    if args[-1][5].dtype == cp.float64:
        return restriction_matrix_2d_kernel_double(*args)
    elif args[-1][5].dtype == cp.float32:
        return restriction_matrix_2d_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



restriction_matrix_3d_kernel_double = _cuda_module.get_function("restriction_matrix_3d_kernel<double>")
restriction_matrix_3d_kernel_float = _cuda_module.get_function("restriction_matrix_3d_kernel<float>")

def restriction_matrix_3d_kernel(*args):
    if args[-1][6].dtype == cp.float64:
        return restriction_matrix_3d_kernel_double(*args)
    elif args[-1][6].dtype == cp.float32:
        return restriction_matrix_3d_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



prolongation_matrix_2d_kernel_double = _cuda_module.get_function("prolongation_matrix_2d_kernel<double>")
prolongation_matrix_2d_kernel_float = _cuda_module.get_function("prolongation_matrix_2d_kernel<float>")

def prolongation_matrix_2d_kernel(*args):
    if args[-1][5].dtype == cp.float64:
        return prolongation_matrix_2d_kernel_double(*args)
    elif args[-1][5].dtype == cp.float32:
        return prolongation_matrix_2d_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



prolongation_matrix_3d_kernel_double = _cuda_module.get_function("prolongation_matrix_3d_kernel<double>")
prolongation_matrix_3d_kernel_float = _cuda_module.get_function("prolongation_matrix_3d_kernel<float>")

def prolongation_matrix_3d_kernel(*args):
    if args[-1][6].dtype == cp.float64:
        return prolongation_matrix_3d_kernel_double(*args)
    elif args[-1][6].dtype == cp.float32:
        return prolongation_matrix_3d_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



generate_nodes_2d_double = _cuda_module.get_function("generate_nodes_2d<double>")
generate_nodes_2d_float = _cuda_module.get_function("generate_nodes_2d<float>")

def generate_nodes_2d(*args):
    if args[-1][0].dtype == cp.float64:
        return generate_nodes_2d_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return generate_nodes_2d_float(*args)
    else:
        raise ValueError("Unsupported dtype")



generate_nodes_3d_double = _cuda_module.get_function("generate_nodes_3d<double>")
generate_nodes_3d_float = _cuda_module.get_function("generate_nodes_3d<float>")

def generate_nodes_3d(*args):
    if args[-1][0].dtype == cp.float64:
        return generate_nodes_3d_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return generate_nodes_3d_float(*args)
    else:
        raise ValueError("Unsupported dtype")



apply_filter_2D_kernel_double = _cuda_module.get_function("apply_filter_2D_kernel<double>")
apply_filter_2D_kernel_float = _cuda_module.get_function("apply_filter_2D_kernel<float>")

def apply_filter_2D_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return apply_filter_2D_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return apply_filter_2D_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



apply_filter_3D_kernel_double = _cuda_module.get_function("apply_filter_3D_kernel<double>")
apply_filter_3D_kernel_float = _cuda_module.get_function("apply_filter_3D_kernel<float>")

def apply_filter_3D_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return apply_filter_3D_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return apply_filter_3D_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



get_filter_2D_weights_kernel_double = _cuda_module.get_function("get_filter_2D_weights_kernel<double>")
get_filter_2D_weights_kernel_float = _cuda_module.get_function("get_filter_2D_weights_kernel<float>")

def get_filter_2D_weights_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return get_filter_2D_weights_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return get_filter_2D_weights_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")



get_filter_3D_weights_kernel_double = _cuda_module.get_function("get_filter_3D_weights_kernel<double>")
get_filter_3D_weights_kernel_float = _cuda_module.get_function("get_filter_3D_weights_kernel<float>")

def get_filter_3D_weights_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return get_filter_3D_weights_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return get_filter_3D_weights_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")


apply_filter_2D_transpose_kernel_double = _cuda_module.get_function("apply_filter_2D_transpose_kernel<double>")
apply_filter_2D_transpose_kernel_float = _cuda_module.get_function("apply_filter_2D_transpose_kernel<float>")

def apply_filter_2D_transpose_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return apply_filter_2D_transpose_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return apply_filter_2D_transpose_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")


apply_filter_3D_transpose_kernel_double = _cuda_module.get_function("apply_filter_3D_transpose_kernel<double>")
apply_filter_3D_transpose_kernel_float = _cuda_module.get_function("apply_filter_3D_transpose_kernel<float>")

def apply_filter_3D_transpose_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return apply_filter_3D_transpose_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return apply_filter_3D_transpose_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")


FEA_Integrals_node_basis_parallel_cuda_kernel_double = _cuda_module.get_function("FEA_Integrals_node_basis_parallel_cuda_kernel<double>")
FEA_Integrals_node_basis_parallel_cuda_kernel_float = _cuda_module.get_function("FEA_Integrals_node_basis_parallel_cuda_kernel<float>")

def FEA_Integrals_node_basis_parallel_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return FEA_Integrals_node_basis_parallel_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return FEA_Integrals_node_basis_parallel_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")

FEA_Integrals_node_basis_parallel_full_cuda_kernel_double = _cuda_module.get_function("FEA_Integrals_node_basis_parallel_full_cuda_kernel<double>")
FEA_Integrals_node_basis_parallel_full_cuda_kernel_float = _cuda_module.get_function("FEA_Integrals_node_basis_parallel_full_cuda_kernel<float>")

def FEA_Integrals_node_basis_parallel_full_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return FEA_Integrals_node_basis_parallel_full_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return FEA_Integrals_node_basis_parallel_full_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")

FEA_Integrals_node_basis_parallel_flat_cuda_kernel_double = _cuda_module.get_function("FEA_Integrals_node_basis_parallel_flat_cuda_kernel<double>")
FEA_Integrals_node_basis_parallel_flat_cuda_kernel_float = _cuda_module.get_function("FEA_Integrals_node_basis_parallel_flat_cuda_kernel<float>")

def FEA_Integrals_node_basis_parallel_flat_cuda_kernel(*args):
    if args[-1][0].dtype == cp.float64:
        return FEA_Integrals_node_basis_parallel_flat_cuda_kernel_double(*args)
    elif args[-1][0].dtype == cp.float32:
        return FEA_Integrals_node_basis_parallel_flat_cuda_kernel_float(*args)
    else:
        raise ValueError("Unsupported dtype")

matmat_node_basis_nnz_per_row_kernel_code = '''
extern "C" __global__
void matmat_node_basis_nnz_per_row_kernel(int* elements_flat, int* el_ids, int* sorter, 
    int* node_ids, int n_nodes, int dof, int elements_size, int n_col, 
    int* Bp, int* Bj, int elem_flat_size, int* nnz_per_row) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;

    int rolling_index[max_nnz];
    
    for (int j = 0; j < max_nnz; j++) {
        rolling_index[j] = -1;
    }
    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = elem_flat_size;
    }

    int n_elements = en-st;
    for (int j = 0; j < n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ids_j * elements_size;
        
        for (int l = 0; l < elements_size*dof; l++) {
            int jj = elements_flat[start + l/dof]*dof + l%dof;
            
            for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++) {
                int k = Bj[kk];
                for (int ll = 0; ll < max_nnz; ll++) {
                    if (rolling_index[ll] == k) {
                        break;
                    } else if (rolling_index[ll] == -1) {
                        rolling_index[ll] = k;
                        for (int d = 0; d < dof; d++) {
                            atomicAdd(&nnz_per_row[i*dof + d], 1);
                        }
                        break;
                        if (ll == max_nnz-1){
                            printf("Error: Exceeded maximum number of nonzeros per row");
                            return;
                        }
                    }
                }
            }
        }
    }
}
'''

csrcsr_matmat_nnz_per_row_kernel_code = '''
extern "C" __global__
void csrcsr_matmat_nnz_per_row_kernel(int* Ap, int* Aj, int* Bp, int* Bj, int n_row, int n_col, int* nnz_per_row) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_row) return;

    int rolling_index[max_nnz];
    
    for (int j = 0; j < max_nnz; j++) {
        rolling_index[j] = -1;
    }
    
    int st = Ap[i];
    int en = Ap[i+1];
    
    for (int j = st; j < en; j++) {
        int jj = Aj[j];
        for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++) {
            int k = Bj[kk];
            for (int ll = 0; ll < max_nnz; ll++) {
                if (rolling_index[ll] == k) {
                    break;
                } else if (rolling_index[ll] == -1) {
                    rolling_index[ll] = k;
                    atomicAdd(&nnz_per_row[i], 1);
                    break;
                    if (ll == max_nnz-1){
                        printf("Error: Exceeded maximum number of nonzeros per row");
                        return;
                    }
                }
            }
        }
    }
}
'''




@lru_cache(maxsize=None)
def get_csrcsr_matmat_nnz_per_row_kernel(max_nnz):
    return cp.RawKernel(csrcsr_matmat_nnz_per_row_kernel_code.replace('max_nnz', str(max_nnz)), 'csrcsr_matmat_nnz_per_row_kernel')

def csrcsr_matmat_nnz_per_row_kernel(B,T,A, max_nnz):
    kernel = get_csrcsr_matmat_nnz_per_row_kernel(max_nnz)
    kernel(B, T, A)


matmat_node_basis_flat_nnz_per_row_kernel_code = '''
extern "C" __global__
void matmat_node_basis_flat_nnz_per_row_kernel(int* elements_flat, int* elements_ptr, 
    int* el_ids, int* sorter, int* node_ids, int n_nodes, int dof, int n_col,
    int* Bp, int* Bj, int elem_flat_size, int* nnz_per_row) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;

    int rolling_index[max_nnz];
    
    for (int j = 0; j < max_nnz; j++) {
        rolling_index[j] = -1;
    }
    
    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = elem_flat_size;
    }
    
    int n_elements = en-st;
    
    for (int j = 0; j < n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ptr[elements_ids_j];
        int end = elements_ptr[elements_ids_j+1];
        int size = end - start;
        
        for (int l = 0; l < size*dof; l++) {
            int jj = elements_flat[start + l/dof]*dof + l%dof;
            
            for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++) {
                int k = Bj[kk];
                for (int ll = 0; ll < max_nnz; ll++) {
                    if (rolling_index[ll] == k) {
                        break;
                    } else if (rolling_index[ll] == -1) {
                        rolling_index[ll] = k;
                        for (int d = 0; d < dof; d++) {
                            atomicAdd(&nnz_per_row[i*dof + d], 1);
                        }
                        break;
                    }
                    if (ll == max_nnz-1){
                        printf("Error: Exceeded maximum number of nonzeros per row");
                        return;
                    }
                }
            }
        }
    }
}
'''

@lru_cache(maxsize=None)
def get_matmat_node_basis_nnz_per_row_kernel(max_nnz):
    return cp.RawKernel(
        matmat_node_basis_nnz_per_row_kernel_code.replace('max_nnz', str(max_nnz)), 
        'matmat_node_basis_nnz_per_row_kernel'
    )

def matmat_node_basis_nnz_per_row_kernel(B,T,A, max_nnz):
    kernel = get_matmat_node_basis_nnz_per_row_kernel(max_nnz)
    kernel(B, T, A)


@lru_cache(maxsize=None)
def get_matmat_node_basis_flat_nnz_per_row_kernel(max_nnz):
    return cp.RawKernel(
        matmat_node_basis_flat_nnz_per_row_kernel_code.replace('max_nnz', str(max_nnz)), 
        'matmat_node_basis_flat_nnz_per_row_kernel'
    )

def matmat_node_basis_flat_nnz_per_row_kernel(B,T,A, max_nnz):
    kernel = get_matmat_node_basis_flat_nnz_per_row_kernel(max_nnz)
    kernel(B, T, A)



matmat_node_basis_nnz_per_row_wcon_kernel_code = '''
extern "C" __global__
void matmat_node_basis_nnz_per_row_wcon_kernel(int* elements_flat, int* el_ids, int* sorter, 
    int* node_ids, int n_nodes, int dof, int elements_size, int n_col, 
    int* Bp, int* Bj, int elem_flat_size, int* nnz_per_row, bool* con_map) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;

    int rolling_index[max_nnz];
    
    for (int j = 0; j < max_nnz; j++) {
        rolling_index[j] = -1;
    }
    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = elem_flat_size;
    }

    int n_elements = en-st;
    for (int j = 0; j < n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ids_j * elements_size;
        
        for (int l = 0; l < elements_size*dof; l++) {
            int jj = elements_flat[start + l/dof]*dof + l%dof;
            
            if (con_map[jj] == true) continue;
            
            for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++) {
                int k = Bj[kk];
                for (int ll = 0; ll < max_nnz; ll++) {
                    if (rolling_index[ll] == k) {
                        break;
                    } else if (rolling_index[ll] == -1) {
                        rolling_index[ll] = k;
                        for (int d = 0; d < dof; d++) {
                            atomicAdd(&nnz_per_row[i*dof + d], 1);
                        }
                        break;
                        if (ll == max_nnz-1){
                            printf("Error: Exceeded maximum number of nonzeros per row");
                            return;
                        }
                    }
                }
            }
        }
    }
    
    for (int d = 0; d < dof; d++) {
        if (con_map[i*dof+d] == true) {
            nnz_per_row[i*dof+d] = Bp[i*dof+d+1] - Bp[i*dof+d];
        }
    }
}
'''

matmat_node_basis_flat_nnz_per_row_wcon_kernel_code = '''
extern "C" __global__
void matmat_node_basis_flat_nnz_per_row_kernel(int* elements_flat, int* elements_ptr, 
    int* el_ids, int* sorter, int* node_ids, int n_nodes, int dof, int n_col,
    int* Bp, int* Bj, int elem_flat_size, int* nnz_per_row) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;

    int rolling_index[max_nnz];
    
    for (int j = 0; j < max_nnz; j++) {
        rolling_index[j] = -1;
    }
    
    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = elem_flat_size;
    }
    
    int n_elements = en-st;
    
    for (int j = 0; j < n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ptr[elements_ids_j];
        int end = elements_ptr[elements_ids_j+1];
        int size = end - start;
        
        for (int l = 0; l < size*dof; l++) {
            int jj = elements_flat[start + l/dof]*dof + l%dof;
            
            if (con_map[jj] == true) continue;
            
            for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++) {
                int k = Bj[kk];
                for (int ll = 0; ll < max_nnz; ll++) {
                    if (rolling_index[ll] == k) {
                        break;
                    } else if (rolling_index[ll] == -1) {
                        rolling_index[ll] = k;
                        for (int d = 0; d < dof; d++) {
                            atomicAdd(&nnz_per_row[i*dof + d], 1);
                        }
                        break;
                    }
                    if (ll == max_nnz-1){
                        printf("Error: Exceeded maximum number of nonzeros per row");
                        return;
                    }
                }
            }
        }
    }
    
    for (int d = 0; d < dof; d++) {
        if (con_map[i*dof+d] == true) {
            nnz_per_row[i*dof+d] = Bp[i*dof+d+1] - Bp[i*dof+d];
        }
    }
}
'''

@lru_cache(maxsize=None)
def get_matmat_node_basis_nnz_per_row_wcon_kernel(max_nnz):
    return cp.RawKernel(
        matmat_node_basis_nnz_per_row_wcon_kernel_code.replace('max_nnz', str(max_nnz)), 
        'matmat_node_basis_nnz_per_row_wcon_kernel'
    )

def matmat_node_basis_nnz_per_row_wcon_kernel(B,T,A, max_nnz):
    kernel = get_matmat_node_basis_nnz_per_row_wcon_kernel(max_nnz)
    kernel(B, T, A)

@lru_cache(maxsize=None)
def get_matmat_node_basis_flat_nnz_per_row_wcon_kernel(max_nnz):
    return cp.RawKernel(
        matmat_node_basis_flat_nnz_per_row_wcon_kernel_code.replace('max_nnz', str(max_nnz)), 
        'matmat_node_basis_flat_nnz_per_row_kernel'
    )

def matmat_node_basis_flat_nnz_per_row_wcon_kernel(B,T,A, max_nnz):
    kernel = get_matmat_node_basis_flat_nnz_per_row_wcon_kernel(max_nnz)
    kernel(B, T, A)


element_2d_kernel_code = '''
extern "C" __global__
void generate_elements_2d(int* elements, int nx, int ny) {
    int counter = blockDim.x * blockIdx.x + threadIdx.x;
    int nel_x = nx - 1;
    int nel_y = ny - 1;
    int num_elem = nel_x * nel_y;
    
    if (counter < num_elem) {
        int i = counter / nel_y;  // i varies slowest
        int j = counter % nel_y;  // j varies fastest
        
        elements[counter * 4] = j * nx + i;
        elements[counter * 4 + 1] = j * nx + i + 1;
        elements[counter * 4 + 2] = (j + 1) * nx + i + 1;
        elements[counter * 4 + 3] = (j + 1) * nx + i;
    }
}
'''

element_3d_kernel_code = '''
extern "C" __global__
void generate_elements_3d(int* elements, int nx, int ny, int nz) {
    int counter = blockDim.x * blockIdx.x + threadIdx.x;
    int nel_x = nx - 1;
    int nel_y = ny - 1;
    int nel_z = nz - 1;
    int num_elem = nel_x * nel_y * nel_z;
    
    if (counter < num_elem) {
        int i = counter / (nel_y * nel_z);  // i varies slowest
        int tmp = counter % (nel_y * nel_z);
        int j = tmp / nel_z;                // j varies middle
        int k = tmp % nel_z;                // k varies fastest
        
        int idx = counter * 8;
        elements[idx] = nz * i + nz * nx * j + k;
        elements[idx + 1] = nz * (i + 1) + nz * nx * j + k;
        elements[idx + 2] = nz * (i + 1) + nz * nx * (j + 1) + k;
        elements[idx + 3] = nz * i + nz * nx * (j + 1) + k;
        elements[idx + 4] = nz * i + nz * nx * j + k + 1;
        elements[idx + 5] = nz * (i + 1) + nz * nx * j + k + 1;
        elements[idx + 6] = nz * (i + 1) + nz * nx * (j + 1) + k + 1;
        elements[idx + 7] = nz * i + nz * nx * (j + 1) + k + 1;
    }
}
'''

generate_elements_2d_kernel = cp.RawKernel(element_2d_kernel_code, 'generate_elements_2d')
generate_elements_3d_kernel = cp.RawKernel(element_3d_kernel_code, 'generate_elements_3d')
