
template<typename T> __global__
void pop_constraints_vec_cuda_kernel(int* cons, int n_cons, T* vec, T* buffer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cons) return;
    
    buffer[i] = vec[cons[i]];
    vec[cons[i]] = 0;
}

template<typename T> __global__
void recover_constraints_vec_cuda_kernel(int* cons, int n_cons, T* vec, T* buffer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cons) return;
    
    vec[cons[i]] = buffer[i];
}

template<typename T> __global__
void pop_constraints_cuda_kernel(int* Ap, int* Aj, T* Ax, int* cons, int n_cons, int nnz, int* ptr, T* data, int* indices) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cons) return;
    
    int start = Ap[cons[i]];
    int end = Ap[cons[i]+1];
    int counter = 0;
    for (int j = ptr[cons[i]]; j < ptr[cons[i]+1]; j++) {
        data[j] = Ax[start+counter];
        indices[j] = Aj[start+counter];
        Ax[start+counter] = 0;
        counter++;
    }
}

template<typename T> __global__
void recover_constraints_cuda_kernel(int* Ap, int* Aj, T* Ax, int* cons, int n_cons, T* data, int* indices, int* ptr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cons) return;
    
    int start = Ap[cons[i]];
    int end = Ap[cons[i]+1];
    int counter = 0;
    for (int j = ptr[cons[i]]; j < ptr[cons[i]+1]; j++) {
        Ax[start+counter] = data[j];
        Aj[start+counter] = indices[j];
        counter++;
    }
}

template<typename T> __global__
void mat_vec_node_basis_parallel_flat_cuda_kernel(T* K_flat, int* elements_flat, int* K_ptr, int* elements_ptr, int* el_ids, T* weights, int* sorter, int* node_ids, int n_nodes, T* vec, int dof, int elem_flat_size, T* out) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_nodes) {
        int st, en;
        if (i<n_nodes-1) {
            st = node_ids[i];
            en = node_ids[i+1];
        } else {
            st = node_ids[i];
            en = elem_flat_size;
        }

        for (int k=0; k<dof; k++) {
            out[i*dof+k] = 0;
        }
        
        int n_elements = en-st;
        T val = 0.0;
        for (int j=0; j<n_elements; j++) {
            int e_ind_j = sorter[st+j];
            int elements_ids_j = el_ids[e_ind_j];
            int start = elements_ptr[elements_ids_j];
            int end = elements_ptr[elements_ids_j+1];
            int size = end - start;
            T weight = weights[elements_ids_j];
            
            int relative_dof = e_ind_j - start;
            int k_start = K_ptr[elements_ids_j] + relative_dof * dof * size * dof;
            
            for (int k=0; k<dof; k++) {
                val = 0.0;
                for (int l=0; l<size*dof; l++) {
                    // atomicAdd(&out[i*dof+k], K_flat[k_start + k*size*dof + l] * vec[elements_flat[start+l/dof]*dof + l % dof] * weight);
                    val += K_flat[k_start + k*size*dof + l] * vec[elements_flat[start+l/dof]*dof + l % dof];
                }
                out[i*dof+k] += val * weight;
            }
        }
    } 
}

template<typename T> __global__
void mat_vec_node_basis_parallel_full_cuda_kernel(T* K_flat, int* elements_flat, int* el_ids, T* weights, int* sorter, int* node_ids, int n_nodes, T* vec, int dof, int elements_size, int elem_flat_size, T* out) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_nodes) {
        int st, en;
        if (i<n_nodes-1) {
            st = node_ids[i];
            en = node_ids[i+1];
        } else {
            st = node_ids[i];
            en = elem_flat_size;
        }
        
        for (int k=0; k<dof; k++) {
            out[i*dof+k] = 0;
        }

        int n_elements = en-st;
        T val = 0.0;
        for (int j=0; j<n_elements; j++) {
            int e_ind_j = sorter[st+j];
            int elements_ids_j = el_ids[e_ind_j];
            int start = elements_ids_j * elements_size;
            int end = start + elements_size;
            T weight = weights[elements_ids_j];
            
            int relative_dof = e_ind_j - start;
            int k_start = elements_ids_j * elements_size * dof * elements_size * dof + relative_dof * dof * elements_size * dof;
            
            for (int k=0; k<dof; k++) {
                val = 0.0;
                for (int l=0; l<elements_size*dof; l++) {
                    // atomicAdd(&out[i*dof+k], K_flat[k_start + k*elements_size*dof + l] * vec[elements_flat[start+l/dof]*dof + l % dof] * weight);
                    val += K_flat[k_start + k*elements_size*dof + l] * vec[elements_flat[start+l/dof]*dof + l % dof];
                }
                out[i*dof+k] += val * weight;
            }
        }
    } 
}

template<typename T> __global__
void mat_vec_node_basis_parallel_cuda_kernel(T* K_flat, int* elements_flat, int* el_ids, T* weights, int* sorter, int* node_ids, int n_nodes, T* vec, int dof, int elements_size, int elem_flat_size, T* out) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_nodes) {
        int st, en;
        if (i<n_nodes-1) {
            st = node_ids[i];
            en = node_ids[i+1];
        } else {
            st = node_ids[i];
            en = elem_flat_size;
        }
        
        for (int k=0; k<dof; k++) {
            out[i*dof+k] = 0;
        }

        int n_elements = en-st;
        T val = 0.0;
        for (int j=0; j<n_elements; j++) {
            int e_ind_j = sorter[st+j];
            int elements_ids_j = el_ids[e_ind_j];
            int start = elements_ids_j * elements_size;
            int end = start + elements_size;
            T weight = weights[elements_ids_j];
            
            int relative_dof = e_ind_j - start;
            int k_start = relative_dof * dof * elements_size * dof;
            
            for (int k=0; k<dof; k++) {
                val = 0.0;
                for (int l=0; l<elements_size*dof; l++) {
                    // atomicAdd(&out[i*dof+k], K_flat[k_start + k*elements_size*dof + l] * vec[elements_flat[start+l/dof]*dof + l % dof] * weight);
                    val += K_flat[k_start + k*elements_size*dof + l] * vec[elements_flat[start+l/dof]*dof + l % dof];
                }
                out[i*dof+k] += val * weight;
            }
        }
    } 
}

template<typename T> __global__
void process_dk_kernel_cuda(T* K_flat, int* elements_flat, T* U, int dof, int elements_size, int nel, T* out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < nel){
        out[i] = 0;
        T val = 0;
        for(int j=0; j<dof*elements_size; j++){
            val = 0;
            for(int k=0; k<dof*elements_size; k++){
                val += K_flat[j*dof*elements_size + k] * U[elements_flat[i*elements_size + k/dof]*dof + k % dof];
            }
            out[i] -= val * U[elements_flat[i*elements_size + j/dof]*dof + j % dof];
        }
    }
}

template<typename T> __global__
void process_dk_full_kernel_cuda(T* K_flat, int* elements_flat, T* U, int dof, int elements_size, int nel, T* out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < nel){
        out[i] = 0;
        T val = 0;
        int k_start = dof*elements_size*dof*elements_size * i;
        for(int j=0; j<dof*elements_size; j++){
            val = 0;
            for(int k=0; k<dof*elements_size; k++){
                val += K_flat[k_start + j*dof*elements_size + k] * U[elements_flat[i*elements_size + k/dof]*dof + k % dof];
            }
            out[i] -= val * U[elements_flat[i*elements_size + j/dof]*dof + j % dof];
        }
    }
}

template<typename T> __global__
void process_dk_flat_kernel_cuda(T* K_flat, int* elements_flat, int* K_ptr, int* elements_ptr, T* U, int dof, int nel, T* out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < nel){
        out[i] = 0;
        T val = 0;
        int k_start = K_ptr[i];
        int size = elements_ptr[i+1] - elements_ptr[i];
        
        for(int j=0; j<dof*size; j++){
            val = 0;
            for(int k=0; k<dof*size; k++){
                val += K_flat[k_start + j*dof*size + k] * U[elements_flat[elements_ptr[i] + k/dof]*dof + k % dof];
            }
            out[i] -= val * U[elements_flat[elements_ptr[i] + j/dof]*dof + j % dof];
        }
    }
}

// Restriction kernel for 3D
template<typename T> __global__
void restriction_3d_kernel(T* v_fine, T* v_coarse, int nx_fine, int ny_fine, int nz_fine, 
                   int nx_coarse, int ny_coarse, int nz_coarse, int dof) {
    int idx_coarse = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_coarse >= nx_coarse * ny_coarse * nz_coarse) return;

    // Convert linear index to 3D coordinates
    int k_coarse = idx_coarse % nz_coarse;
    int tmp = idx_coarse / nz_coarse;
    int i_coarse = tmp % nx_coarse;
    int j_coarse = tmp / nx_coarse;

    // Corresponding fine grid coordinates
    int i_fine = i_coarse * 2;
    int j_fine = j_coarse * 2;
    int k_fine = k_coarse * 2;

    // T total_weight = weight_sums[idx_coarse];

    int boundaries = 0;

    if (i_coarse == 0 || i_coarse == nx_coarse-1) boundaries++;
    if (j_coarse == 0 || j_coarse == ny_coarse-1) boundaries++;
    if (k_coarse == 0 || k_coarse == nz_coarse-1) boundaries++;

    T total_weight;

    if (boundaries == 3) total_weight = 3.375f;
    else if (boundaries == 2) total_weight = 4.5f;
    else if (boundaries == 1) total_weight = 6.0f;
    else total_weight = 8.0f;

    // For each DOF
    for (int d = 0; d < dof; d++) {
        T value = 0.0f;

        // Loop over neighborhood in fine grid
        for (int di = -1; di <= 1; di++) {
            int i_neighbor = i_fine + di;
            if (i_neighbor < 0 || i_neighbor >= nx_fine) continue;

            for (int dj = -1; dj <= 1; dj++) {
                int j_neighbor = j_fine + dj;
                if (j_neighbor < 0 || j_neighbor >= ny_fine) continue;

                for (int dk = -1; dk <= 1; dk++) {
                    int k_neighbor = k_fine + dk;
                    if (k_neighbor < 0 || k_neighbor >= nz_fine) continue;

                    // Calculate distance and weight
                    int distance = abs(di) + abs(dj) + abs(dk);
                    if (distance > 3) continue;

                    T weight;
                    if (distance == 0) weight = 1.0f;
                    else if (distance == 1) weight = 0.5f;
                    else if (distance == 2) weight = 0.25f;
                    else weight = 0.125f;

                    // Calculate fine grid linear index
                    int idx_fine = k_neighbor + nz_fine * i_neighbor + nz_fine * nx_fine * j_neighbor;
                    value += weight * v_fine[idx_fine * dof + d];
                }
            }
        }
        // Normalize and store result
        if (total_weight > 0) {
            v_coarse[idx_coarse * dof + d] = value / total_weight;
        }
    }
}

// Restriction kernel for 2D
template<typename T> __global__
void restriction_2d_kernel(T* v_fine, T* v_coarse, int nx_fine, int ny_fine,
                   int nx_coarse, int ny_coarse, int dof) {
    int idx_coarse = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_coarse >= nx_coarse * ny_coarse) return;

    // Convert linear index to 2D coordinates
    int i_coarse = idx_coarse % nx_coarse;
    int j_coarse = idx_coarse / nx_coarse;

    // Corresponding fine grid coordinates
    int i_fine = i_coarse * 2;
    int j_fine = j_coarse * 2;

    int boundaries = 0;

    if (i_coarse == 0 || i_coarse == nx_coarse-1) boundaries++;
    if (j_coarse == 0 || j_coarse == ny_coarse-1) boundaries++;

    T total_weight;

    if (boundaries == 2) total_weight = 2.25f;
    else if (boundaries == 1) total_weight = 3.0f;
    else total_weight = 4.0f;

    // For each DOF
    for (int d = 0; d < dof; d++) {
        T value = 0.0f;

        // Loop over neighborhood in fine grid
        for (int di = -1; di <= 1; di++) {
            int i_neighbor = i_fine + di;
            if (i_neighbor < 0 || i_neighbor >= nx_fine) continue;

            for (int dj = -1; dj <= 1; dj++) {
                int j_neighbor = j_fine + dj;
                if (j_neighbor < 0 || j_neighbor >= ny_fine) continue;

                // Calculate distance and weight
                int distance = abs(di) + abs(dj);
                if (distance > 2) continue;

                T weight;
                if (distance == 0) weight = 1.0f;
                else if (distance == 1) weight = 0.5f;
                else weight = 0.25f;

                // Calculate fine grid linear index
                int idx_fine = i_neighbor + nx_fine * j_neighbor;
                value += weight * v_fine[idx_fine * dof + d];
            }
        }

        // Normalize and store result
        if (total_weight > 0) {
            v_coarse[idx_coarse * dof + d] = value / total_weight;
        }
    }
}

// Prolongation kernel for 3D
template<typename T> __global__
void prolongation_3d_kernel(T* v_coarse, T* v_fine, int nx_fine, int ny_fine, int nz_fine,
                    int nx_coarse, int ny_coarse, int nz_coarse, int dof) {
    int idx_fine = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_fine >= nx_fine * ny_fine * nz_fine) return;

    // Convert linear index to 3D coordinates
    int k = idx_fine % nz_fine;
    int tmp = idx_fine / nz_fine;
    int i = tmp % nx_fine;
    int j = tmp / nx_fine;

    // Get coarse grid indices
    int i_coarse = i / 2;
    int j_coarse = j / 2;
    int k_coarse = k / 2;

    // For each DOF
    for (int d = 0; d < dof; d++) {
        T value = 0.0f;
        int count = 0;

        for (int ci = i_coarse; ci <= i_coarse + (i % 2); ci++) {
            if (ci >= nx_coarse) continue;
            
            for (int cj = j_coarse; cj <= j_coarse + (j % 2); cj++) {
                if (cj >= ny_coarse) continue;
                
                for (int ck = k_coarse; ck <= k_coarse + (k % 2); ck++) {
                    if (ck >= nz_coarse) continue;

                    int idx_coarse = ck + nz_coarse * ci + nz_coarse * nx_coarse * cj;

                    int boundaries = 0;

                    if (ci == 0 || ci == nx_coarse-1) boundaries++;
                    if (cj == 0 || cj == ny_coarse-1) boundaries++;
                    if (ck == 0 || ck == nz_coarse-1) boundaries++;

                    T total_weight;

                    if (boundaries == 3) total_weight = 3.375f;
                    else if (boundaries == 2) total_weight = 4.5f;
                    else if (boundaries == 1) total_weight = 6.0f;
                    else total_weight = 8.0f;       


                    // T total_weight = weight_sums[idx_coarse];
                    value += v_coarse[idx_coarse * dof + d]/total_weight;
                    count++;
                }
            }
        }

        if (count > 0) {
            v_fine[idx_fine * dof + d] = value / count;
        }
    }
}

// Prolongation kernel for 2D
template<typename T> __global__
void prolongation_2d_kernel(T* v_coarse, T* v_fine, int nx_fine, int ny_fine,
                    int nx_coarse, int ny_coarse, int dof) {
    int idx_fine = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_fine >= nx_fine * ny_fine) return;

    // Convert linear index to 2D coordinates
    int i = idx_fine % nx_fine;
    int j = idx_fine / nx_fine;

    // Get coarse grid indices
    int i_coarse = i / 2;
    int j_coarse = j / 2;

    // For each DOF
    for (int d = 0; d < dof; d++) {
        T value = 0.0f;
        int count = 0;

        for (int ci = i_coarse; ci <= i_coarse + (i % 2); ci++) {
            if (ci >= nx_coarse) continue;
            
            for (int cj = j_coarse; cj <= j_coarse + (j % 2); cj++) {
                if (cj >= ny_coarse) continue;

                int idx_coarse = ci + nx_coarse * cj;

                int boundaries = 0;
                
                T total_weight;

                if (ci == 0 || ci == nx_coarse-1) boundaries++;
                if (cj == 0 || cj == ny_coarse-1) boundaries++;

                if (boundaries == 2) total_weight = 2.25f;
                else if (boundaries == 1) total_weight = 3.0f;
                else total_weight = 4.0f;

                value += v_coarse[idx_coarse * dof + d]/total_weight;
                count++;
            }
        }

        if (count > 0) {
            v_fine[idx_fine * dof + d] = value / count;
        }
    }
}

template<typename T> __global__
void csrcsr_matmat_kernel(int* Ap, int* Aj, T* Ax, int* Bp, int* Bj, T* Bx, int n_row, int* Cp, int* Cj, T* Cx) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_row) return;
    
    for (int j = Ap[i]; j < Ap[i+1]; j++) {
        int jj = Aj[j];
        for (int k = Bp[jj]; k < Bp[jj+1]; k++) {
            int kk = Bj[k];
            for (int l = Cp[i]; l < Cp[i+1]; l++) {
                if (Cj[l] == kk) {
                    atomicAdd(&Cx[l], Ax[j] * Bx[k]);
                    break;
                } else if (Cj[l] == -1) {
                    Cj[l] = kk;
                    atomicAdd(&Cx[l], Ax[j] * Bx[k]);
                    break;
                }
            }
        }
    }
}

template<typename T> __global__
void matmat_node_basis_parallel_kernel(T* K_single, int* elements_flat, int* el_ids, T* weights, int* sorter, int* node_ids, int n_nodes, int dof, int elements_size,int* Bp, int* Bj, T* Bx, int* Cp, int* Cj, T* Cx, int max_elem_size) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_nodes) return;

	int start_ = Cp[i*dof];
	int end_ = Cp[i*dof+1];
	int length = end_ - start_;

    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = max_elem_size;
    }

    int n_elements = en-st;
    for (int j = 0; j < n_elements; j++) {
		int e_ind_j = sorter[st+j];
		int elements_ids_j = el_ids[e_ind_j];
		int start = elements_ids_j * elements_size;
		int end = start + elements_size;
		int relative_dof = e_ind_j - start;
        
		for (int l = 0; l < elements_size*dof; l++) {
			int jj = elements_flat[start + l/dof]*dof + l % dof;
			for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++) {
				int k = Bj[kk];
				for (int ll = start_; ll < end_; ll++) {
					if (Cj[ll] == k) {
						for (int d = 0; d < dof; d++) {
							atomicAdd(&Cx[ll + length * d], K_single[(relative_dof*dof+d) * elements_size * dof + l] * Bx[kk] * weights[elements_ids_j]);
						}
						break;
					} else if (Cj[ll] == -1) {
						for (int d = 0; d < dof; d++) {
							Cj[ll + length * d] = k;
							atomicAdd(&Cx[ll + length * d], K_single[(relative_dof*dof+d) * elements_size * dof + l] * Bx[kk] * weights[elements_ids_j]);
						}
						break;
					}
				}
			}
		}
	}
}

template<typename T> __global__
void matmat_node_basis_full_parallel_kernel(T* Ks, int* elements_flat, int* el_ids, T* weights, 
    int* sorter, int* node_ids, int n_nodes, int dof, int elements_size, 
    int* Bp, int* Bj, T* Bx, int* Cp, int* Cj, T* Cx, int max_elem_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;

    int start_ = Cp[i*dof];
    int end_ = Cp[i*dof+1];
    int length = end_ - start_;

    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = max_elem_size;
    }

    int n_elements = en-st;
    for (int j = 0; j < n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ids_j * elements_size;
        int end = start + elements_size;
        T weight = weights[elements_ids_j];
        int relative_dof = e_ind_j - start;
        
        for (int l = 0; l < elements_size*dof; l++) {
            int jj = elements_flat[start + l/dof]*dof + l % dof;
            for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++) {
                int k = Bj[kk];
                for (int ll = start_; ll < end_; ll++) {
                    if (Cj[ll] == k) {
                        for (int d = 0; d < dof; d++) {
                            atomicAdd(&Cx[ll + length * d], 
                                Ks[elements_ids_j * elements_size * dof * elements_size * dof + 
                                (relative_dof*dof+d) * elements_size * dof + l] * 
                                Bx[kk] * weight);
                        }
                        break;
                    } else if (Cj[ll] == -1) {
                        for (int d = 0; d < dof; d++) {
                            Cj[ll + length * d] = k;
                            atomicAdd(&Cx[ll + length * d], 
                                Ks[elements_ids_j * elements_size * dof * elements_size * dof + 
                                (relative_dof*dof+d) * elements_size * dof + l] * 
                                Bx[kk] * weight);
                        }
                        break;
                    }
                }
            }
        }
    }
}

template<typename T> __global__
void matmat_node_basis_flat_parallel_kernel(T* K_flat, int* elements_flat, int* K_ptr, 
    int* elements_ptr, int* el_ids, T* weights, int* sorter, int* node_ids, 
    int n_nodes, int dof, int* Bp, int* Bj, T* Bx, int* Cp, int* Cj, T* Cx, 
    int max_elem_size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;

    int start_ = Cp[i*dof];
    int end_ = Cp[i*dof+1];
    int length = end_ - start_;

    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = max_elem_size;
    }

    int n_elements = en-st;
    for (int j = 0; j < n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ptr[elements_ids_j];
        int end = elements_ptr[elements_ids_j+1];
        int size = end - start;
        T weight = weights[elements_ids_j];
        
        int relative_dof = e_ind_j - start;
        int k_start = K_ptr[elements_ids_j] + relative_dof * dof * size * dof;
        
        for (int l = 0; l < size*dof; l++) {
            int jj = elements_flat[start + l/dof]*dof + l % dof;
            for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++) {
                int k = Bj[kk];
                for (int ll = start_; ll < end_; ll++) {
                    if (Cj[ll] == k) {
                        for (int d = 0; d < dof; d++) {
                            atomicAdd(&Cx[ll + length * d], 
                                K_flat[k_start + d*size*dof + l] * Bx[kk] * weight);
                        }
                        break;
                    } else if (Cj[ll] == -1) {
                        for (int d = 0; d < dof; d++) {
                            Cj[ll + length * d] = k;
                            atomicAdd(&Cx[ll + length * d], 
                                K_flat[k_start + d*size*dof + l] * Bx[kk] * weight);
                        }
                        break;
                    }
                }
            }
        }
    }
}

template<typename T> __global__
void mat_vec_node_basis_parallel_flat_wcon_cuda_kernel(T* K_flat, int* elements_flat, int* K_ptr, int* elements_ptr, int* el_ids, T* weights, int* sorter, int* node_ids, int n_nodes, T* vec, int dof, int elem_flat_size, T* out, bool* con_map) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_nodes) {
        
        int st, en;
        if (i<n_nodes-1) {
            st = node_ids[i];
            en = node_ids[i+1];
        } else {
            st = node_ids[i];
            en = elem_flat_size;
        }

        for (int k=0; k<dof; k++) {
            if (con_map[i*dof+k] == true) {
                out[i*dof+k] = vec[i*dof+k];
            }
            else{
                out[i*dof+k] = 0;
            }
        }
        
        int n_elements = en-st;
        T val = 0.0;
        for (int j=0; j<n_elements; j++) {
            int e_ind_j = sorter[st+j];
            int elements_ids_j = el_ids[e_ind_j];
            int start = elements_ptr[elements_ids_j];
            int end = elements_ptr[elements_ids_j+1];
            int size = end - start;
            T weight = weights[elements_ids_j];
            
            int relative_dof = e_ind_j - start;
            int k_start = K_ptr[elements_ids_j] + relative_dof * dof * size * dof;
            
            for (int k=0; k<dof; k++) {
                if (con_map[i*dof+k] == true) {
                    continue;
                }
                val = 0.0;
                for (int l=0; l<size*dof; l++) {
                    int kk = elements_flat[start+l/dof]*dof + l % dof;
                    if (con_map[kk] == true) {
                        continue;
                    }
                    // atomicAdd(&out[i*dof+k], K_flat[k_start + k*size*dof + l] * vec[kk] * weight);
                    val += K_flat[k_start + k*size*dof + l] * vec[kk];
                }
                out[i*dof+k] += val * weight;
            }
        }
    } 
}

template<typename T> __global__
void mat_vec_node_basis_parallel_full_wcon_cuda_kernel(T* K_flat, int* elements_flat, int* el_ids, T* weights, int* sorter, int* node_ids, int n_nodes, T* vec, int dof, int elements_size, int elem_flat_size, T* out, bool* con_map) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_nodes) {
        
        int st, en;
        if (i<n_nodes-1) {
            st = node_ids[i];
            en = node_ids[i+1];
        } else {
            st = node_ids[i];
            en = elem_flat_size;
        }
        
        for (int k=0; k<dof; k++) {
            if (con_map[i*dof+k] == true) {
                out[i*dof+k] = vec[i*dof+k];
            }
            else{
                out[i*dof+k] = 0;
            }
        }

        int n_elements = en-st;
        T val = 0.0;
        for (int j=0; j<n_elements; j++) {
            int e_ind_j = sorter[st+j];
            int elements_ids_j = el_ids[e_ind_j];
            int start = elements_ids_j * elements_size;
            int end = start + elements_size;
            T weight = weights[elements_ids_j];
            
            int relative_dof = e_ind_j - start;
            int k_start = elements_ids_j * elements_size * dof * elements_size * dof + relative_dof * dof * elements_size * dof;
            
            for (int k=0; k<dof; k++) {
                if (con_map[i*dof+k] == true) {
                    continue;
                }
                val = 0.0;
                for (int l=0; l<elements_size*dof; l++) {
                    int kk = elements_flat[start+l/dof]*dof + l % dof;
                    if (con_map[kk] == true) {
                        continue;
                    }
                    val += K_flat[k_start + k*elements_size*dof + l] * vec[kk];
                    // atomicAdd(&out[i*dof+k], K_flat[k_start + k*elements_size*dof + l] * vec[kk] * weight);
                }
                out[i*dof+k] += val * weight;
            }
        }
    } 
}

template<typename T> __global__
void mat_vec_node_basis_parallel_wcon_cuda_kernel(T* K_flat, int* elements_flat, int* el_ids, T* weights, int* sorter, int* node_ids, int n_nodes, T* vec, int dof, int elements_size, int elem_flat_size, T* out, bool* con_map) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_nodes) {
        int st, en;
        if (i<n_nodes-1) {
            st = node_ids[i];
            en = node_ids[i+1];
        } else {
            st = node_ids[i];
            en = elem_flat_size;
        }
        
        for (int k=0; k<dof; k++) {
            if (con_map[i*dof+k] == true) {
                out[i*dof+k] = vec[i*dof+k];
            }
            else{
                out[i*dof+k] = 0;
            }
        }

        int n_elements = en-st;
        T val = 0.0;
        for (int j=0; j<n_elements; j++) {
            int e_ind_j = sorter[st+j];
            int elements_ids_j = el_ids[e_ind_j];
            int start = elements_ids_j * elements_size;
            int end = start + elements_size;
            T weight = weights[elements_ids_j];
            
            int relative_dof = e_ind_j - start;
            int k_start = relative_dof * dof * elements_size * dof;
            
            for (int k=0; k<dof; k++) {
                if (con_map[i*dof+k] == true) {
                    continue;
                }
                val = 0.0;
                for (int l=0; l<elements_size*dof; l++) {
                    int kk = elements_flat[start+l/dof]*dof + l % dof;
                    if (con_map[kk] == true) {
                        continue;
                    }
                    val += K_flat[k_start + k*elements_size*dof + l] * vec[kk];
                }
                out[i*dof+k] += val * weight;
            }
        }
    }
}

template<typename T> __global__
void matmat_node_basis_full_parallel_wcon_kernel(T* Ks, int* elements_flat, int* el_ids, T* weights, 
    int* sorter, int* node_ids, int n_nodes, int dof, int elements_size, 
    int* Bp, int* Bj, T* Bx, int* Cp, int* Cj, T* Cx, int max_elem_size, bool* con_map) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;

    int start_ = Cp[i*dof];
    int end_ = Cp[i*dof+1];
    int length = end_ - start_;

    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = max_elem_size;
    }

    int n_elements = en-st;
    for (int j = 0; j < n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ids_j * elements_size;
        int end = start + elements_size;
        T weight = weights[elements_ids_j];
        int relative_dof = e_ind_j - start;
        
        for (int l = 0; l < elements_size*dof; l++) {
            int jj = elements_flat[start + l/dof]*dof + l % dof;
            if (con_map[jj] == true) {
                continue;
            }
			for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++) {
				int k = Bj[kk];
                for (int d = 0; d < dof; d++) {
                    if (con_map[i*dof+d] == true) {
                        continue;
                    }
                    for (int ll = start_; ll < end_; ll++) {
                        if (Cj[ll + length * d] == k) {
                            atomicAdd(&Cx[ll + length * d], 
                                    Ks[elements_ids_j * elements_size * dof * elements_size * dof + 
                                    (relative_dof*dof+d) * elements_size * dof + l] * 
                                    Bx[kk] * weight);
                            break;
                        } else if (Cj[ll + length * d] == -1) {
                            Cj[ll + length * d] = k;
                            atomicAdd(&Cx[ll + length * d], 
                                    Ks[elements_ids_j * elements_size * dof * elements_size * dof + 
                                    (relative_dof*dof+d) * elements_size * dof + l] * 
                                    Bx[kk] * weight);
                            break;
                        }
                    }
                }
			}
        }
    }
    
    for (int d = 0; d < dof; d++) {
        if (con_map[i*dof+d] == true) {
            int jj = i*dof+d;
            int count = 0;
            for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++){
                int k = Bj[kk];
                Cj[start_ + length * d + count] = k;
                Cx[start_ + length * d + count] = Bx[kk];
                count++;
            }
            while (count < length) {
                Cj[start_ + length * d + count] = 0;
                Cx[start_ + length * d + count] = 0;
                count++;
            }
        }
    }
}

template<typename T> __global__
void matmat_node_basis_flat_parallel_wcon_kernel(T* K_flat, int* elements_flat, int* K_ptr, 
    int* elements_ptr, int* el_ids, T* weights, int* sorter, int* node_ids, 
    int n_nodes, int dof, int* Bp, int* Bj, T* Bx, int* Cp, int* Cj, T* Cx, 
    int max_elem_size, bool* con_map) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;

    int start_ = Cp[i*dof];
    int end_ = Cp[i*dof+1];
    int length = end_ - start_;

    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = max_elem_size;
    }

    int n_elements = en-st;
    for (int j = 0; j < n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ptr[elements_ids_j];
        int end = elements_ptr[elements_ids_j+1];
        int size = end - start;
        T weight = weights[elements_ids_j];
        
        int relative_dof = e_ind_j - start;
        int k_start = K_ptr[elements_ids_j] + relative_dof * dof * size * dof;
        
        for (int l = 0; l < size*dof; l++) {
            int jj = elements_flat[start + l/dof]*dof + l % dof;
            if (con_map[jj] == true) {
                continue;
            }
			for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++) {
				int k = Bj[kk];
                for (int d = 0; d < dof; d++) {
                    if (con_map[i*dof+d] == true) {
                        continue;
                    }
                    for (int ll = start_; ll < end_; ll++) {
                        if (Cj[ll + length * d] == k) {
                            atomicAdd(&Cx[ll + length * d], 
                                    K_flat[k_start + d*size*dof + l] * Bx[kk] * weight);
                            break;
                        } else if (Cj[ll + length * d] == -1) {
                            Cj[ll + length * d] = k;
                            atomicAdd(&Cx[ll + length * d], 
                                    K_flat[k_start + d*size*dof + l] * Bx[kk] * weight);
                            break;
                        }
                    }
                }
			}
        }
    }
    
    for (int d = 0; d < dof; d++) {
        if (con_map[i*dof+d] == true) {
            int jj = i*dof+d;
            int count = 0;
            for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++){
                int k = Bj[kk];
                Cj[start_ + length * d + count] = k;
                Cx[start_ + length * d + count] = Bx[kk];
                count++;
            }
            while (count < length) {
                Cj[start_ + length * d + count] = 0;
                Cx[start_ + length * d + count] = 0;
                count++;
            }
        }
    }
}

template<typename T> __global__
void matmat_node_basis_parallel_wcon_kernel(T* K_single, int* elements_flat, int* el_ids, T* weights,
    int* sorter, int* node_ids, int n_nodes, int dof, int elements_size,
    int* Bp, int* Bj, T* Bx, int* Cp, int* Cj, T* Cx, int max_elem_size, bool* con_map) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_nodes) return;

	int start_ = Cp[i*dof];
	int end_ = Cp[i*dof+1];
	int length = end_ - start_;

    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = max_elem_size;
    }

    int n_elements = en-st;
    for (int j = 0; j < n_elements; j++) {
		int e_ind_j = sorter[st+j];
		int elements_ids_j = el_ids[e_ind_j];
		int start = elements_ids_j * elements_size;
		int end = start + elements_size;
		int relative_dof = e_ind_j - start;
        
		for (int l = 0; l < elements_size*dof; l++) {
			int jj = elements_flat[start + l/dof]*dof + l % dof;
            if (con_map[jj] == true) {
                continue;
            }
			for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++) {
				int k = Bj[kk];
                for (int d = 0; d < dof; d++) {
                    if (con_map[i*dof+d] == true) {
                        continue;
                    }
                    for (int ll = start_; ll < end_; ll++) {
                        if (Cj[ll + length * d] == k) {
                            atomicAdd(&Cx[ll + length * d], K_single[(relative_dof*dof+d) * elements_size * dof + l] * Bx[kk] * weights[elements_ids_j]);
                            break;
                        } else if (Cj[ll + length * d] == -1) {
                            Cj[ll + length * d] = k;
                            atomicAdd(&Cx[ll + length * d], K_single[(relative_dof*dof+d) * elements_size * dof + l] * Bx[kk] * weights[elements_ids_j]);
                            break;
                        }
                    }
                }
			}
		}
	}
 
    for (int d = 0; d < dof; d++) {
        if (con_map[i*dof+d] == true) {
            int jj = i*dof+d;
            int count = 0;
            for (int kk = Bp[jj]; kk < Bp[jj+1]; kk++){
                int k = Bj[kk];
                Cj[start_ + length * d + count] = k;
                Cx[start_ + length * d + count] = Bx[kk];
                count++;
            }
            while (count < length) {
                Cj[start_ + length * d + count] = 0;
                Cx[start_ + length * d + count] = 0;
                count++;
            }
        }
    }
}

template<typename T> __global__
void get_diagonal_node_basis_cuda_kernel(T* K_single, int* elements_flat, int* el_ids, T* weights, 
    int* sorter, int* node_ids, int n_nodes, int dof, int elements_size, bool* cons_map, T* diag, int max_elem_size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    
    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = max_elem_size;
    }

    for (int k = 0; k < dof; k++) {
        if (cons_map[i*dof+k]) {
            diag[i*dof+k] = 1;
        } else {
            diag[i*dof+k] = 0;
        }
    }
    
    int n_elements = en-st;
    
    for (int j = 0; j < n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ids_j * elements_size;
        int end = start + elements_size;
        T weight = weights[elements_ids_j];
        int relative_dof = e_ind_j - start;
        
        for (int k = 0; k < dof; k++) {
            if (cons_map[i*dof+k]) {
                continue;
            }
            for (int l = 0; l < elements_size*dof; l++) {
                if (elements_flat[start+l/dof]*dof+l%dof == i*dof+k) {
                    atomicAdd(&diag[i*dof+k], K_single[relative_dof*dof+k + (elements_size*dof)*l] * weight);
                }
            }
        }
    }
}

template<typename T> __global__
void get_diagonal_node_basis_full_cuda_kernel(T* Ks, int* elements_flat, int* el_ids, T* weights, 
    int* sorter, int* node_ids, int n_nodes, int dof, int elements_size, bool* cons_map, T* diag, int max_elem_size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    
    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = max_elem_size;
    }

    for (int k = 0; k < dof; k++) {
        if (cons_map[i*dof+k]) {
            diag[i*dof+k] = 1;
        } else {
            diag[i*dof+k] = 0;
        }
    }
    
    int n_elements = en-st;
    
    for (int j = 0; j < n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ids_j * elements_size;
        int end = start + elements_size;
        T weight = weights[elements_ids_j];
        int relative_dof = e_ind_j - start;
        
        for (int k = 0; k < dof; k++) {
            if (cons_map[i*dof+k]) {
                continue;
            }
            for (int l = 0; l < elements_size*dof; l++) {
                if (elements_flat[start+l/dof]*dof+l%dof == i*dof+k) {
                    atomicAdd(&diag[i*dof+k], Ks[elements_ids_j * (elements_size*dof*elements_size*dof) + 
                        (relative_dof*dof+k) * (elements_size*dof) + l] * weight);
                }
            }
        }
    }
}

template<typename T> __global__
void get_diagonal_node_basis_flat_cuda_kernel(T* K_flat, int* elements_flat, int* K_ptr, 
    int* elements_ptr, int* el_ids, T* weights, int* sorter, int* node_ids, 
    int n_nodes, int dof, bool* cons_map, T* diag, int max_elem_size) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nodes) return;
    
    int st, en;
    if (i < n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = max_elem_size;
    }
    
    int n_elements = en-st;

    for (int k = 0; k < dof; k++) {
        if (cons_map[i*dof+k]) {
            diag[i*dof+k] = 1;
        } else {
            diag[i*dof+k] = 0;
        }
    }
    
    for (int j = 0; j < n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ptr[elements_ids_j];
        int end = elements_ptr[elements_ids_j+1];
        int size = end - start;
        T weight = weights[elements_ids_j];
        
        int relative_dof = e_ind_j - start;
        int k_start = K_ptr[elements_ids_j] + relative_dof * dof * size * dof;
        
        for (int k = 0; k < dof; k++) {
            if (cons_map[i*dof+k]) {
                continue;
            }
            for (int l = 0; l < size*dof; l++) {
                if (elements_flat[start+l/dof]*dof+l%dof == i*dof+k) {
                    atomicAdd(&diag[i*dof+k], K_flat[k_start + k*size*dof + l] * weight);
                }
            }
        }
    }
}

template<typename T> __global__
void restriction_matrix_2d_kernel(int nx, int ny, int dof, int* Cp, int* Cj, T* Cx) {
    int idx_coarse = blockIdx.x * blockDim.x + threadIdx.x;
    
    int nx_fine = nx + 1;
    int ny_fine = ny + 1;
    int nx_coarse = nx/2 + 1;
    int ny_coarse = ny/2 + 1;
    int n_coarse_nodes = nx_coarse * ny_coarse;
    
    if (idx_coarse >= n_coarse_nodes) return;
    
    int i_coarse = idx_coarse % nx_coarse;
    int j_coarse = idx_coarse / nx_coarse;
    
    // Corresponding fine grid coordinates
    int i_fine = i_coarse * 2;
    int j_fine = j_coarse * 2;
    
    int boundaries = 0;

    if (i_coarse == 0 || i_coarse == nx_coarse-1) boundaries++;
    if (j_coarse == 0 || j_coarse == ny_coarse-1) boundaries++;

    T total_weight;

    if (boundaries == 2) total_weight = 2.25f;
    else if (boundaries == 1) total_weight = 3.0f;
    else total_weight = 4.0f;

    for (int d = 0; d < dof; d++) {
        int counter = 0;
        int row_start = Cp[idx_coarse * dof + d];
        
        // Loop over neighborhood in fine grid
        for (int di = -1; di <= 1; di++) {
            int i_neighbor = i_fine + di;
            if (i_neighbor < 0 || i_neighbor >= nx_fine) continue;
            
            for (int dj = -1; dj <= 1; dj++) {
                int j_neighbor = j_fine + dj;
                if (j_neighbor < 0 || j_neighbor >= ny_fine) continue;
                
                // Calculate distance and weight
                int distance = abs(di) + abs(dj);
                if (distance > 2) continue;
                
                T weight = 1.0f;
                if (distance == 1) weight = 0.5f;
                else if (distance == 2) weight = 0.25f;
                
                // Calculate fine grid linear index
                int idx_fine = i_neighbor + nx_fine * j_neighbor;
                
                Cj[row_start + counter] = idx_fine * dof + d;
                Cx[row_start + counter] = weight/total_weight;
                counter++;
            }
        }
    }
}

template<typename T> __global__
void restriction_matrix_3d_kernel(int nx, int ny, int nz, int dof, int* Cp, int* Cj, T* Cx) {
    int idx_coarse = blockIdx.x * blockDim.x + threadIdx.x;
    
    int nx_fine = nx + 1;
    int ny_fine = ny + 1;
    int nz_fine = nz + 1;
    int nx_coarse = nx/2 + 1;
    int ny_coarse = ny/2 + 1;
    int nz_coarse = nz/2 + 1;
    int n_coarse_nodes = nx_coarse * ny_coarse * nz_coarse;
    
    if (idx_coarse >= n_coarse_nodes) return;
    
    int k_coarse = idx_coarse % nz_coarse;
    int tmp = idx_coarse / nz_coarse;
    int i_coarse = tmp % nx_coarse;
    int j_coarse = tmp / nx_coarse;
    
    // Corresponding fine grid coordinates
    int i_fine = i_coarse * 2;
    int j_fine = j_coarse * 2;
    int k_fine = k_coarse * 2;

    // T total_weight = weight_sums[idx_coarse];

    int boundaries = 0;

    if (i_coarse == 0 || i_coarse == nx_coarse-1) boundaries++;
    if (j_coarse == 0 || j_coarse == ny_coarse-1) boundaries++;
    if (k_coarse == 0 || k_coarse == nz_coarse-1) boundaries++;

    T total_weight;

    if (boundaries == 3) total_weight = 3.375f;
    else if (boundaries == 2) total_weight = 4.5f;
    else if (boundaries == 1) total_weight = 6.0f;
    else total_weight = 8.0f;
    
    
    for (int d = 0; d < dof; d++) {
        int counter = 0;
        int row_start = Cp[idx_coarse * dof + d];
        
        // Loop over neighborhood in fine grid
        for (int di = -1; di <= 1; di++) {
            int i_neighbor = i_fine + di;
            if (i_neighbor < 0 || i_neighbor >= nx_fine) continue;
            
            for (int dj = -1; dj <= 1; dj++) {
                int j_neighbor = j_fine + dj;
                if (j_neighbor < 0 || j_neighbor >= ny_fine) continue;
                
                for (int dk = -1; dk <= 1; dk++) {
                    int k_neighbor = k_fine + dk;
                    if (k_neighbor < 0 || k_neighbor >= nz_fine) continue;
                    
                    // Calculate distance and weight
                    int distance = abs(di) + abs(dj) + abs(dk);
                    if (distance > 3) continue;
                    
                    T weight = 1.0f;
                    if (distance == 1) weight = 0.5f;
                    else if (distance == 2) weight = 0.25f;
                    else if (distance == 3) weight = 0.125f;
                    
                    // Calculate fine grid linear index
                    int idx_fine = k_neighbor + nz_fine * i_neighbor + nz_fine * nx_fine * j_neighbor;
                    
                    Cj[row_start + counter] = idx_fine * dof + d;
                    Cx[row_start + counter] = weight/total_weight;
                    counter++;
                }
            }
        }
    }
}

template<typename T> __global__
void prolongation_matrix_2d_kernel(int nx, int ny, int dof, int* Cp, int* Cj, T* Cx) {
    int idx_fine = blockIdx.x * blockDim.x + threadIdx.x;
    
    int nx_fine = nx + 1;
    int ny_fine = ny + 1;
    int nx_coarse = nx/2 + 1;
    int ny_coarse = ny/2 + 1;
    int n_fine_nodes = nx_fine * ny_fine;
    
    if (idx_fine >= n_fine_nodes) return;
    
    int i = idx_fine % nx_fine;
    int j = idx_fine / nx_fine;
    
    // Get coarse grid indices
    int i_coarse = i / 2;
    int j_coarse = j / 2;
    
    for (int d = 0; d < dof; d++) {
        int count = 0;
        int counter = 0;
        int row_start = Cp[idx_fine * dof + d];
        
        // Loop over contributing coarse nodes
        for (int ci = i_coarse; ci <= i_coarse + (i % 2); ci++) {
            if (ci >= nx_coarse) continue;
            
            for (int cj = j_coarse; cj <= j_coarse + (j % 2); cj++) {
                if (cj >= ny_coarse) continue;
                
                count++;
                int idx_coarse = ci + nx_coarse * cj;

                int boundaries = 0;

                if (ci == 0 || ci == nx_coarse-1) boundaries++;
                if (cj == 0 || cj == ny_coarse-1) boundaries++;

                T total_weight;

                if (boundaries == 2) total_weight = 2.25f;
                else if (boundaries == 1) total_weight = 3.0f;
                else total_weight = 4.0f;
                
                Cj[row_start + counter] = idx_coarse * dof + d;
                Cx[row_start + counter] = 1.0f/total_weight;
                counter++;
            }
        }
        
        // Normalize weights
        if (count > 0) {
            T weight = 1.0f / count;
            for (int k = 0; k < counter; k++) {
                Cx[row_start + k] *= weight;
            }
        }
    }
}

template<typename T> __global__
void prolongation_matrix_3d_kernel(int nx, int ny, int nz, int dof, int* Cp, int* Cj, T* Cx) {
    int idx_fine = blockIdx.x * blockDim.x + threadIdx.x;
    
    int nx_fine = nx + 1;
    int ny_fine = ny + 1;
    int nz_fine = nz + 1;
    int nx_coarse = nx/2 + 1;
    int ny_coarse = ny/2 + 1;
    int nz_coarse = nz/2 + 1;
    int n_fine_nodes = nx_fine * ny_fine * nz_fine;
    
    if (idx_fine >= n_fine_nodes) return;
    
    int k = idx_fine % nz_fine;
    int tmp = idx_fine / nz_fine;
    int i = tmp % nx_fine;
    int j = tmp / nx_fine;
    
    // Get coarse grid indices
    int i_coarse = i / 2;
    int j_coarse = j / 2;
    int k_coarse = k / 2;
    
    for (int d = 0; d < dof; d++) {
        int count = 0;
        int counter = 0;
        int row_start = Cp[idx_fine * dof + d];
        
        // Loop over contributing coarse nodes
        for (int ci = i_coarse; ci <= i_coarse + (i % 2); ci++) {
            if (ci >= nx_coarse) continue;
            
            for (int cj = j_coarse; cj <= j_coarse + (j % 2); cj++) {
                if (cj >= ny_coarse) continue;
                
                for (int ck = k_coarse; ck <= k_coarse + (k % 2); ck++) {
                    if (ck >= nz_coarse) continue;
                    
                    count++;
                    int idx_coarse = ck + nz_coarse * ci + nz_coarse * nx_coarse * cj;
                    // T total_weight = weight_sums[idx_coarse];

                    int boundaries = 0;

                    if (ci == 0 || ci == nx_coarse-1) boundaries++;
                    if (cj == 0 || cj == ny_coarse-1) boundaries++;
                    if (ck == 0 || ck == nz_coarse-1) boundaries++;

                    T total_weight;

                    if (boundaries == 3) total_weight = 3.375f;
                    else if (boundaries == 2) total_weight = 4.5f;
                    else if (boundaries == 1) total_weight = 6.0f;
                    else total_weight = 8.0f;   

                    Cj[row_start + counter] = idx_coarse * dof + d;
                    Cx[row_start + counter] = 1.0f/total_weight;
                    counter++;
                }
            }
        }
        
        // Normalize weights
        if (count > 0) {
            T weight = 1.0f / count;
            for (int k = 0; k < counter; k++) {
                Cx[row_start + k] *= weight;
            }
        }
    }
}

template<typename T> __global__
void generate_nodes_2d(T* node_positions, T L, T H, int nx, int ny) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_nodes = nx * ny;
    
    if (idx < total_nodes) {
        int i = idx % nx;  // x index
        int j = idx / nx;  // y index
        
        T dx = L / (nx - 1);
        T dy = H / (ny - 1);
        
        node_positions[idx * 2] = i * dx;      // x coordinate
        node_positions[idx * 2 + 1] = j * dy;  // y coordinate
    }
}

template<typename T> __global__
void generate_nodes_3d(T* node_positions, T L, T H, T W, int nx, int ny, int nz) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_nodes = nx * ny * nz;
    
    if (idx < total_nodes) {
        int i = (idx / nz) % nx;     // x index
        int j = idx / (nx * nz);     // y index
        int k = idx % nz;            // z index
        
        T dx = L / (nx - 1);
        T dy = H / (ny - 1);
        T dz = W / (nz - 1);
        
        node_positions[idx * 3] = i * dx;      // x coordinate
        node_positions[idx * 3 + 1] = j * dy;  // y coordinate
        node_positions[idx * 3 + 2] = k * dz;  // z coordinate
    }
}


template<typename T> __global__
void apply_filter_2D_kernel(T* v_in, T* v_out, int nelx, int nely, int* offsets, T* weights, int offsets_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nelx * nely) {
        int ix = i / nely;
        int iy = i % nely;

        T sum_weighted_values = 0.0;
        T sum_weights = 0.0;
        
        for (int k = 0; k < offsets_len; k++) {
            int dx = offsets[2 * k];
            int dy = offsets[2 * k + 1];
            int ix_n = ix + dx;
            int iy_n = iy + dy;
            
            if (ix_n >= 0 && ix_n < nelx && iy_n >= 0 && iy_n < nely) {
                int neighbor_idx = ix_n * nely + iy_n;
                T weight = weights[k];
                sum_weighted_values += weight * v_in[neighbor_idx];
                sum_weights += weight;
            }
        }
        
        v_out[i] = sum_weights > 0 ? sum_weighted_values / sum_weights : 0.0f;
    }
}

template<typename T> __global__
void apply_filter_3D_kernel(T* v_in, T* v_out, int nelx, int nely, int nelz, int* offsets, T* weights, int offsets_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nelx * nely * nelz) {
        int ix = i / (nely * nelz);
        int iy = (i / nelz) % nely;
        int iz = i % nelz;

        T sum_weighted_values = 0.0;
        T sum_weights = 0.0;
        
        for (int k = 0; k < offsets_len; k++) {
            int dx = offsets[3 * k];
            int dy = offsets[3 * k + 1];
            int dz = offsets[3 * k + 2];
            int ix_n = ix + dx;
            int iy_n = iy + dy;
            int iz_n = iz + dz;
            
            if (ix_n >= 0 && ix_n < nelx && iy_n >= 0 && iy_n < nely && iz_n >= 0 && iz_n < nelz) {
                int neighbor_idx = ix_n * nely * nelz + iy_n * nelz + iz_n;
                T weight = weights[k];
                sum_weighted_values += weight * v_in[neighbor_idx];
                sum_weights += weight;
            }
        }
        
        v_out[i] = sum_weights > 0 ? sum_weighted_values / sum_weights : 0.0f;
    }
}

template<typename T> __global__
void get_filter_2D_weights_kernel(T* normalization, int nelx, int nely, int* offsets, T* weights, int offsets_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nelx * nely) return;
    
    int ix = i / nely;
    int iy = i % nely;
    
    T sum_weights = 0.0f;
    for (int k = 0; k < offsets_len; k++) {
        int dx = offsets[2 * k];
        int dy = offsets[2 * k + 1];
        int ix_n = ix + dx;
        int iy_n = iy + dy;
        
        if (ix_n >= 0 && ix_n < nelx && iy_n >= 0 && iy_n < nely) {
            sum_weights += weights[k];
        }
    }
    normalization[i] = sum_weights;
}

template<typename T> __global__
void get_filter_3D_weights_kernel(T* normalization, int nelx, int nely, int nelz, int* offsets, T* weights, int offsets_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nelx * nely * nelz) return;
    
    int ix = i / (nely * nelz);
    int iy = (i / nelz) % nely;
    int iz = i % nelz;
    
    T sum_weights = 0.0f;
    for (int k = 0; k < offsets_len; k++) {
        int dx = offsets[3 * k];
        int dy = offsets[3 * k + 1];
        int dz = offsets[3 * k + 2];
        int ix_n = ix + dx;
        int iy_n = iy + dy;
        int iz_n = iz + dz;
        
        if (ix_n >= 0 && ix_n < nelx && iy_n >= 0 && iy_n < nely && iz_n >= 0 && iz_n < nelz) {
            sum_weights += weights[k];
        }
    }
    normalization[i] = sum_weights;
}

template<typename T> __global__
void apply_filter_2D_transpose_kernel(T* v_in, T* v_out, int nelx, int nely, int* offsets, T* weights, T* normalization, int offsets_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nelx * nely) return;
    
    int ix = i / nely;
    int iy = i % nely;
    
    T sum_value = 0.0f;
    for (int k = 0; k < offsets_len; k++) {
        int dx = offsets[2 * k];
        int dy = offsets[2 * k + 1];
        int jx = ix + dx;
        int jy = iy + dy;
        
        if (jx >= 0 && jx < nelx && jy >= 0 && jy < nely) {
            int j_idx = jx * nely + jy;
            sum_value += (weights[k] / normalization[j_idx]) * v_in[j_idx];
        }
    }
    v_out[i] = sum_value;
}

template<typename T> __global__
void apply_filter_3D_transpose_kernel(T* v_in, T* v_out, int nelx, int nely, int nelz, int* offsets, T* weights, T* normalization, int offsets_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nelx * nely * nelz) return;
    
    int ix = i / (nely * nelz);
    int iy = (i / nelz) % nely;
    int iz = i % nelz;
    
    T sum_value = 0.0f;
    for (int k = 0; k < offsets_len; k++) {
        int dx = offsets[3 * k];
        int dy = offsets[3 * k + 1];
        int dz = offsets[3 * k + 2];
        int jx = ix + dx;
        int jy = iy + dy;
        int jz = iz + dz;
        
        if (jx >= 0 && jx < nelx && jy >= 0 && jy < nely && jz >= 0 && jz < nelz) {
            int j_idx = jx * nely * nelz + jy * nelz + jz;
            sum_value += (weights[k] / normalization[j_idx]) * v_in[j_idx];
        }
    }
    v_out[i] = sum_value;
}

template<typename T> __global__
void FEA_Integrals_node_basis_parallel_cuda_kernel(T* K_single, T* D_single, T* B_single, int* elements_flat, 
    int nel, T* weights, T* U, int dof, int elements_size, int B_size, T* strain, T* stress, T* strain_energy) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nel) return;

    int start = i * elements_size;
    int end = start + elements_size;
    T weight = weights[i];
    
    // Calculate strain
    for (int j = 0; j < B_size; j++) {
        strain[i * B_size + j] = 0;
        for (int k = 0; k < elements_size * dof; k++) {
            strain[i * B_size + j] += B_single[j * elements_size * dof + k] * 
                U[elements_flat[start + k/dof] * dof + k%dof];
        }
    }
    
    // Calculate stress
    for (int j = 0; j < B_size; j++) {
        stress[i * B_size + j] = 0;
        for (int k = 0; k < B_size; k++) {
            stress[i * B_size + j] += D_single[j * B_size + k] * strain[i * B_size + k];
        }
    }
    
    // Calculate strain energy
    strain_energy[i] = 0;
    for (int j = 0; j < elements_size * dof; j++) {
        T val = 0;
        for (int k = 0; k < elements_size * dof; k++) {
            val += weight * K_single[j * elements_size * dof + k] * 
                U[elements_flat[start + k/dof] * dof + k%dof];
        }
        strain_energy[i] += 0.5 * U[elements_flat[start + j/dof] * dof + j%dof] * val;
    }
}

template<typename T> __global__
void FEA_Integrals_node_basis_parallel_full_cuda_kernel(T* Ks, T* Ds, T* Bs, int* elements_flat,
    int nel, T* weights, T* U, int dof, int elements_size, int B_size, T* strain, T* stress, T* strain_energy) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nel) return;

    int start = i * elements_size;
    int end = start + elements_size;
    T weight = weights[i];
    
    // Calculate strain
    for (int j = 0; j < B_size; j++) {
        strain[i * B_size + j] = 0;
        for (int k = 0; k < elements_size * dof; k++) {
            strain[i * B_size + j] += Bs[i * B_size * elements_size * dof + j * elements_size * dof + k] * 
                U[elements_flat[start + k/dof] * dof + k%dof];
        }
    }
    
    // Calculate stress
    for (int j = 0; j < B_size; j++) {
        stress[i * B_size + j] = 0;
        for (int k = 0; k < B_size; k++) {
            stress[i * B_size + j] += Ds[i * B_size * B_size + j * B_size + k] * strain[i * B_size + k];
        }
    }
    
    // Calculate strain energy
    strain_energy[i] = 0;
    for (int j = 0; j < elements_size * dof; j++) {
        T val = 0;
        for (int k = 0; k < elements_size * dof; k++) {
            val += weight * Ks[i * elements_size * dof * elements_size * dof + j * elements_size * dof + k] * 
                U[elements_flat[start + k/dof] * dof + k%dof];
        }
        strain_energy[i] += 0.5 * U[elements_flat[start + j/dof] * dof + j%dof] * val;
    }
}

template<typename T> __global__
void FEA_Integrals_node_basis_parallel_flat_cuda_kernel(T* K_flat, T* D_flat, T* B_flat, int* elements_flat,
    int* elements_ptr, int* K_ptr, int* B_ptr, int* D_ptr, int nel, T* weights, T* U, int dof, int B_size,
    T* strain, T* stress, T* strain_energy) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nel) return;

    int start = elements_ptr[i];
    int end = elements_ptr[i + 1];
    int size = end - start;
    T weight = weights[i];
    
    // Calculate strain
    for (int j = 0; j < B_size; j++) {
        strain[i * B_size + j] = 0;
        for (int k = 0; k < size * dof; k++) {
            strain[i * B_size + j] += B_flat[B_ptr[i] + j * size * dof + k] * 
                U[elements_flat[start + k/dof] * dof + k%dof];
        }
    }
    
    // Calculate stress
    for (int j = 0; j < B_size; j++) {
        stress[i * B_size + j] = 0;
        for (int k = 0; k < B_size; k++) {
            stress[i * B_size + j] += D_flat[D_ptr[i] + j * B_size + k] * strain[i * B_size + k];
        }
    }
    
    // Calculate strain energy
    strain_energy[i] = 0;
    for (int j = 0; j < size * dof; j++) {
        T val = 0;
        for (int k = 0; k < size * dof; k++) {
            val += weight * K_flat[K_ptr[i] + j * size * dof + k] * 
                U[elements_flat[start + k/dof] * dof + k%dof];
        }
        strain_energy[i] += 0.5 * U[elements_flat[start + j/dof] * dof + j%dof] * val;
    }
}