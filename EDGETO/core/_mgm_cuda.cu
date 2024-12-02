template<typename T> __device__
void get_target_vals(T vals[9], T* K_flat, int idx_fine, int dof, int elements_size, int* el_ids, int* node_ids, int* sorter, int* elements_flat, T* weights, int elem_flat_size, int n_nodes, bool* con_map, int i_fine_target, int j_fine_target, int k_fine_target, int nx_fine, int ny_fine, int nz_fine) {
    
    int i = idx_fine;
    int st, en;
    if (i<n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = elem_flat_size;
    }

    int n_elements = en-st;

    for (int k=0; k<dof; k++){
        if (con_map[idx_fine*dof+k] == true) {
            int fine_node = idx_fine;

            int k_fine = fine_node % nz_fine;
            int tmp = fine_node / nz_fine;
            int i_fine = tmp % nx_fine;
            int j_fine = tmp / nx_fine;

            int di = i_fine - i_fine_target;
            int dj = j_fine - j_fine_target;
            int dk = k_fine - k_fine_target;

            if (di < -1 || di > 1) continue;
            if (dj < -1 || dj > 1) continue;
            if (dk < -1 || dk > 1) continue;

            int distance = abs(di) + abs(dj) + abs(dk);
            if (distance > 3) continue;

            T weight;
            if (distance == 0) weight = 1.0f;
            else if (distance == 1) weight = 0.5f;
            else if (distance == 2) weight = 0.25f;
            else weight = 0.125f;

            vals[k*dof+k] = 1.0f*weight;
        }
    }

    for (int j=0; j<n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ids_j * elements_size;
        T rho = weights[elements_ids_j];
        
        int relative_dof = e_ind_j - start;
        int k_start = relative_dof * dof * elements_size * dof;
        
        for (int l=0; l<elements_size; l++) {
            int fine_node = elements_flat[start+l];

            int k_fine = fine_node % nz_fine;
            int tmp = fine_node / nz_fine;
            int i_fine = tmp % nx_fine;
            int j_fine = tmp / nx_fine;

            int di = i_fine - i_fine_target;
            int dj = j_fine - j_fine_target;
            int dk = k_fine - k_fine_target;

            if (di < -1 || di > 1) continue;
            if (dj < -1 || dj > 1) continue;
            if (dk < -1 || dk > 1) continue;

            int distance = abs(di) + abs(dj) + abs(dk);
            if (distance > 3) continue;

            T weight;
            if (distance == 0) weight = 1.0f;
            else if (distance == 1) weight = 0.5f;
            else if (distance == 2) weight = 0.25f;
            else weight = 0.125f;

            for (int k=0; k<dof; k++) {
                if (con_map[i*dof+k] == true) {
                    continue;
                }
                for(int kk=0; kk<dof; kk++) {
                    if (con_map[fine_node*dof+kk] == true) {
                        continue;
                    }
                    vals[k*dof+kk] += K_flat[k_start + k*elements_size*dof + l*dof + kk] * weight * rho;
                }
            }
        }
    }
}

template<typename T> __global__
void get_restricted_3d_l0_nnz_based(T* K_flat, int nx_fine, int ny_fine, int nz_fine,
                                    int nx_coarse, int ny_coarse, int nz_coarse,
                                    int* Cp, int* Cj, T* Cx,
                                    int dof, int* node_ids, int* el_ids, int* sorter, int* elements_flat,
                                    T* weights, int elements_size, bool* con_map, int elem_flat_size, int n_nodes) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;

    int idx_coarse = thread/27;
    if (idx_coarse >= nx_coarse * ny_coarse * nz_coarse) return;

    int neighbour_idx = thread%27;

    int k_coarse = idx_coarse % nz_coarse;
    int tmp = idx_coarse / nz_coarse;
    int i_coarse = tmp % nx_coarse;
    int j_coarse = tmp / nx_coarse;

    int i_offset = neighbour_idx%3 - 1;
    int j_offset = (neighbour_idx/3)%3 - 1;
    int k_offset = (neighbour_idx/9)%3 - 1;

    if (i_coarse + i_offset < 0 || i_coarse + i_offset >= nx_coarse) return;
    if (j_coarse + j_offset < 0 || j_coarse + j_offset >= ny_coarse) return;
    if (k_coarse + k_offset < 0 || k_coarse + k_offset >= nz_coarse) return;

    int target_coarse = k_coarse + k_offset + nz_coarse * (i_coarse + i_offset) + nz_coarse * nx_coarse * (j_coarse + j_offset);

    int i_fine_target = (i_coarse + i_offset) * 2;
    int j_fine_target = (j_coarse + j_offset) * 2;
    int k_fine_target = (k_coarse + k_offset) * 2;

    int i_fine = i_coarse * 2;
    int j_fine = j_coarse * 2;
    int k_fine = k_coarse * 2;

    int boundaries = 0;

    if (i_coarse == 0 || i_coarse == nx_coarse-1) boundaries++;
    if (j_coarse == 0 || j_coarse == ny_coarse-1) boundaries++;
    if (k_coarse == 0 || k_coarse == nz_coarse-1) boundaries++;

    T total_weight;

    if (boundaries == 3) total_weight = 3.375f;
    else if (boundaries == 2) total_weight = 4.5f;
    else if (boundaries == 1) total_weight = 6.0f;
    else total_weight = 8.0f;

    T total_weight_target;

    boundaries = 0;

    if (i_coarse + i_offset == 0 || i_coarse + i_offset == nx_coarse-1) boundaries++;
    if (j_coarse + j_offset == 0 || j_coarse + j_offset == ny_coarse-1) boundaries++;
    if (k_coarse + k_offset == 0 || k_coarse + k_offset == nz_coarse-1) boundaries++;

    if (boundaries == 3) total_weight_target = 3.375f;
    else if (boundaries == 2) total_weight_target = 4.5f;
    else if (boundaries == 1) total_weight_target = 6.0f;
    else total_weight_target = 8.0f;

    for(int k=0; k<dof; k++){
        int start = Cp[idx_coarse*dof + k];
        for (int kk=0; kk<dof; kk++){
            Cj[start + neighbour_idx*dof + kk] = target_coarse*dof + kk;
        }
    }

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

                T vals[9]={0.0f};
                get_target_vals<T>(vals, K_flat, idx_fine, dof, elements_size, el_ids, node_ids, sorter, elements_flat, weights, elem_flat_size, n_nodes, con_map, i_fine_target, j_fine_target, k_fine_target, nx_fine, ny_fine, nz_fine);

                for (int k=0; k<dof; k++){
                    int start = Cp[idx_coarse*dof + k];
                    for (int kk=0; kk<dof; kk++){
                        Cx[start + neighbour_idx*dof + kk] += weight * vals[k*dof + kk] / total_weight / total_weight_target;
                    }
                }
            }
        }
    }
}
// get_target_vals_l1p<T>(vals, dof, i_fine_target, j_fine_target, k_fine_target, nx_fine, ny_fine, nz_fine, Ap, Aj, Ax);
template<typename T> __device__
void get_target_vals_l1p(T vals[9], int idx_fine, int dof, int i_fine_target, int j_fine_target, int k_fine_target, int nx_fine, int ny_fine, int nz_fine, int* Ap, int* Aj, T* Ax) {

    for(int k=0; k<dof; k++){
        int start = Ap[idx_fine*dof + k];
        int end = Ap[idx_fine*dof + k + 1];
        for (int j=start; j<end; j++){
            int fine_node = Aj[j]/dof;
            int local_dof = Aj[j]%dof;
            
            int k_fine = fine_node % nz_fine;
            int tmp = fine_node / nz_fine;
            int i_fine = tmp % nx_fine;
            int j_fine = tmp / nx_fine;

            int di = i_fine - i_fine_target;
            int dj = j_fine - j_fine_target;
            int dk = k_fine - k_fine_target;

            if (di < -1 || di > 1) continue;
            if (dj < -1 || dj > 1) continue;
            if (dk < -1 || dk > 1) continue;

            int distance = abs(di) + abs(dj) + abs(dk);
            if (distance > 3) continue;

            T weight;
            if (distance == 0) weight = 1.0f;
            else if (distance == 1) weight = 0.5f;
            else if (distance == 2) weight = 0.25f;
            else weight = 0.125f;

            vals[k*dof+local_dof] += Ax[j]*weight;
        }
    }
}

template<typename T> __global__
void get_restricted_3d_l1p_nnz_based(T* Ax, int* Aj, int* Ap, int nx_fine, int ny_fine, int nz_fine,
                                    int nx_coarse, int ny_coarse, int nz_coarse,
                                    int* Cp, int* Cj, T* Cx, int dof) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;

    int idx_coarse = thread/27;
    if (idx_coarse >= nx_coarse * ny_coarse * nz_coarse) return;

    int neighbour_idx = thread%27;

    int k_coarse = idx_coarse % nz_coarse;
    int tmp = idx_coarse / nz_coarse;
    int i_coarse = tmp % nx_coarse;
    int j_coarse = tmp / nx_coarse;

    int i_offset = neighbour_idx%3 - 1;
    int j_offset = (neighbour_idx/3)%3 - 1;
    int k_offset = (neighbour_idx/9)%3 - 1;

    if (i_coarse + i_offset < 0 || i_coarse + i_offset >= nx_coarse) return;
    if (j_coarse + j_offset < 0 || j_coarse + j_offset >= ny_coarse) return;
    if (k_coarse + k_offset < 0 || k_coarse + k_offset >= nz_coarse) return;

    int target_coarse = k_coarse + k_offset + nz_coarse * (i_coarse + i_offset) + nz_coarse * nx_coarse * (j_coarse + j_offset);

    int i_fine_target = (i_coarse + i_offset) * 2;
    int j_fine_target = (j_coarse + j_offset) * 2;
    int k_fine_target = (k_coarse + k_offset) * 2;

    int i_fine = i_coarse * 2;
    int j_fine = j_coarse * 2;
    int k_fine = k_coarse * 2;

    int boundaries = 0;

    if (i_coarse == 0 || i_coarse == nx_coarse-1) boundaries++;
    if (j_coarse == 0 || j_coarse == ny_coarse-1) boundaries++;
    if (k_coarse == 0 || k_coarse == nz_coarse-1) boundaries++;

    T total_weight;

    if (boundaries == 3) total_weight = 3.375f;
    else if (boundaries == 2) total_weight = 4.5f;
    else if (boundaries == 1) total_weight = 6.0f;
    else total_weight = 8.0f;

    T total_weight_target;

    boundaries = 0;

    if (i_coarse + i_offset == 0 || i_coarse + i_offset == nx_coarse-1) boundaries++;
    if (j_coarse + j_offset == 0 || j_coarse + j_offset == ny_coarse-1) boundaries++;
    if (k_coarse + k_offset == 0 || k_coarse + k_offset == nz_coarse-1) boundaries++;

    if (boundaries == 3) total_weight_target = 3.375f;
    else if (boundaries == 2) total_weight_target = 4.5f;
    else if (boundaries == 1) total_weight_target = 6.0f;
    else total_weight_target = 8.0f;

    for(int k=0; k<dof; k++){
        int start = Cp[idx_coarse*dof + k];
        for (int kk=0; kk<dof; kk++){
            Cj[start + neighbour_idx*dof + kk] = target_coarse*dof + kk;
        }
    }

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

                T vals[9]={0.0f};
                get_target_vals_l1p<T>(vals, idx_fine, dof, i_fine_target, j_fine_target, k_fine_target, nx_fine, ny_fine, nz_fine, Ap, Aj, Ax);

                for (int k=0; k<dof; k++){
                    int start = Cp[idx_coarse*dof + k];
                    for (int kk=0; kk<dof; kk++){
                        Cx[start + neighbour_idx*dof + kk] += weight * vals[k*dof + kk] / total_weight / total_weight_target;
                    }
                }
            }
        }
    }
}

template<typename T> __device__
void get_target_vals_2d(T vals[4], T* K_flat, int idx_fine, int dof, int elements_size, int* el_ids, int* node_ids, int* sorter, int* elements_flat, T* weights, int elem_flat_size, int n_nodes, bool* con_map, int i_fine_target, int j_fine_target, int nx_fine, int ny_fine) {
    
    int i = idx_fine;
    int st, en;
    if (i<n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = elem_flat_size;
    }

    int n_elements = en-st;

    for (int k=0; k<dof; k++){
        if (con_map[idx_fine*dof+k] == true) {
            int fine_node = idx_fine;

            int i_fine = fine_node % nx_fine;
            int j_fine = fine_node / nx_fine;

            int di = i_fine - i_fine_target;
            int dj = j_fine - j_fine_target;

            if (di < -1 || di > 1) continue;
            if (dj < -1 || dj > 1) continue;

            int distance = abs(di) + abs(dj);
            if (distance > 2) continue;

            T weight;
            if (distance == 0) weight = 1.0f;
            else if (distance == 1) weight = 0.5f;
            else weight = 0.25f;

            vals[k*dof+k] = 1.0f*weight;
        }
    }

    for (int j=0; j<n_elements; j++) {
        int e_ind_j = sorter[st+j];
        int elements_ids_j = el_ids[e_ind_j];
        int start = elements_ids_j * elements_size;
        T rho = weights[elements_ids_j];
        
        int relative_dof = e_ind_j - start;
        int k_start = relative_dof * dof * elements_size * dof;
        
        for (int l=0; l<elements_size; l++) {
            int fine_node = elements_flat[start+l];

            int i_fine = fine_node % nx_fine;
            int j_fine = fine_node / nx_fine;

            int di = i_fine - i_fine_target;
            int dj = j_fine - j_fine_target;

            if (di < -1 || di > 1) continue;
            if (dj < -1 || dj > 1) continue;

            int distance = abs(di) + abs(dj);
            if (distance > 2) continue;

            T weight;
            if (distance == 0) weight = 1.0f;
            else if (distance == 1) weight = 0.5f;
            else weight = 0.25f;

            for (int k=0; k<dof; k++) {
                if (con_map[i*dof+k] == true) {
                    continue;
                }
                for(int kk=0; kk<dof; kk++) {
                    if (con_map[fine_node*dof+kk] == true) {
                        continue;
                    }
                    vals[k*dof+kk] += K_flat[k_start + k*elements_size*dof + l*dof + kk] * weight * rho;
                }
            }
        }
    }
}

template<typename T> __global__
void get_restricted_2d_l0_nnz_based(T* K_flat, int nx_fine, int ny_fine,
                                    int nx_coarse, int ny_coarse,
                                    int* Cp, int* Cj, T* Cx,
                                    int dof, int* node_ids, int* el_ids, int* sorter, int* elements_flat,
                                    T* weights, int elements_size, bool* con_map, int elem_flat_size, int n_nodes) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;

    int idx_coarse = thread/9;
    if (idx_coarse >= nx_coarse * ny_coarse) return;

    int neighbour_idx = thread%9;

    // Convert linear index to 2D coordinates
    int i_coarse = idx_coarse % nx_coarse;
    int j_coarse = idx_coarse / nx_coarse;

    int i_offset = neighbour_idx%3 - 1;
    int j_offset = (neighbour_idx/3)%3 - 1;

    if (i_coarse + i_offset < 0 || i_coarse + i_offset >= nx_coarse) return;
    if (j_coarse + j_offset < 0 || j_coarse + j_offset >= ny_coarse) return;

    int target_coarse = i_coarse + i_offset + nx_coarse * (j_coarse + j_offset);

    int i_fine_target = (i_coarse + i_offset) * 2;
    int j_fine_target = (j_coarse + j_offset) * 2;

    int i_fine = i_coarse * 2;
    int j_fine = j_coarse * 2;

    int boundaries = 0;

    if (i_coarse == 0 || i_coarse == nx_coarse-1) boundaries++;
    if (j_coarse == 0 || j_coarse == ny_coarse-1) boundaries++;

    T total_weight;

    if (boundaries == 2) total_weight = 2.25f;
    else if (boundaries == 1) total_weight = 3.0f;
    else total_weight = 4.0f;


    T total_weight_target;

    boundaries = 0;

    if (i_coarse + i_offset == 0 || i_coarse + i_offset == nx_coarse-1) boundaries++;
    if (j_coarse + j_offset == 0 || j_coarse + j_offset == ny_coarse-1) boundaries++;

    if (boundaries == 2) total_weight_target = 2.25f;
    else if (boundaries == 1) total_weight_target = 3.0f;
    else total_weight_target = 4.0f;


    for(int k=0; k<dof; k++){
        int start = Cp[idx_coarse*dof + k];
        for (int kk=0; kk<dof; kk++){
            Cj[start + neighbour_idx*dof + kk] = target_coarse*dof + kk;
        }
    }

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

            T vals[4]={0.0f};
            get_target_vals_2d<T>(vals, K_flat, idx_fine, dof, elements_size, el_ids, node_ids, sorter, elements_flat, weights, elem_flat_size, n_nodes, con_map, i_fine_target, j_fine_target, nx_fine, ny_fine);

            for (int k=0; k<dof; k++){
                int start = Cp[idx_coarse*dof + k];
                for (int kk=0; kk<dof; kk++){
                    Cx[start + neighbour_idx*dof + kk] += weight * vals[k*dof + kk] / total_weight / total_weight_target;
                }
            }
        }
    }
}

template<typename T> __device__
void get_target_vals_l1p_2d(T vals[4], int idx_fine, int dof, int i_fine_target, int j_fine_target, int nx_fine, int ny_fine, int* Ap, int* Aj, T* Ax) {

    for(int k=0; k<dof; k++){
        int start = Ap[idx_fine*dof + k];
        int end = Ap[idx_fine*dof + k + 1];
        for (int j=start; j<end; j++){
            int fine_node = Aj[j]/dof;
            int local_dof = Aj[j]%dof;
            
            int i_fine = fine_node % nx_fine;
            int j_fine = fine_node / nx_fine;

            int di = i_fine - i_fine_target;
            int dj = j_fine - j_fine_target;

            if (di < -1 || di > 1) continue;
            if (dj < -1 || dj > 1) continue;

            int distance = abs(di) + abs(dj);
            if (distance > 2) continue;

            T weight;
            if (distance == 0) weight = 1.0f;
            else if (distance == 1) weight = 0.5f;
            else weight = 0.25f;

            vals[k*dof+local_dof] += Ax[j]*weight;
        }
    }
}

template<typename T> __global__
void get_restricted_2d_l1p_nnz_based(T* Ax, int* Aj, int* Ap, int nx_fine, int ny_fine,
                                    int nx_coarse, int ny_coarse,
                                    int* Cp, int* Cj, T* Cx, int dof) {

    int thread = blockIdx.x * blockDim.x + threadIdx.x;

    int idx_coarse = thread/9;
    if (idx_coarse >= nx_coarse * ny_coarse) return;

    int neighbour_idx = thread%9;

    // Convert linear index to 2D coordinates
    int i_coarse = idx_coarse % nx_coarse;
    int j_coarse = idx_coarse / nx_coarse;

    int i_offset = neighbour_idx%3 - 1;
    int j_offset = (neighbour_idx/3)%3 - 1;

    if (i_coarse + i_offset < 0 || i_coarse + i_offset >= nx_coarse) return;
    if (j_coarse + j_offset < 0 || j_coarse + j_offset >= ny_coarse) return;

    int target_coarse = i_coarse + i_offset + nx_coarse * (j_coarse + j_offset);

    int i_fine_target = (i_coarse + i_offset) * 2;
    int j_fine_target = (j_coarse + j_offset) * 2;

    int i_fine = i_coarse * 2;
    int j_fine = j_coarse * 2;

    int boundaries = 0;

    if (i_coarse == 0 || i_coarse == nx_coarse-1) boundaries++;
    if (j_coarse == 0 || j_coarse == ny_coarse-1) boundaries++;

    T total_weight;

    if (boundaries == 2) total_weight = 2.25f;
    else if (boundaries == 1) total_weight = 3.0f;
    else total_weight = 4.0f;


    T total_weight_target;

    boundaries = 0;

    if (i_coarse + i_offset == 0 || i_coarse + i_offset == nx_coarse-1) boundaries++;
    if (j_coarse + j_offset == 0 || j_coarse + j_offset == ny_coarse-1) boundaries++;

    if (boundaries == 2) total_weight_target = 2.25f;
    else if (boundaries == 1) total_weight_target = 3.0f;
    else total_weight_target = 4.0f;


    for(int k=0; k<dof; k++){
        int start = Cp[idx_coarse*dof + k];
        for (int kk=0; kk<dof; kk++){
            Cj[start + neighbour_idx*dof + kk] = target_coarse*dof + kk;
        }
    }

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

            T vals[4]={0.0f};
            get_target_vals_l1p_2d<T>(vals, idx_fine, dof, i_fine_target, j_fine_target, nx_fine, ny_fine, Ap, Aj, Ax);

            for (int k=0; k<dof; k++){
                int start = Cp[idx_coarse*dof + k];
                for (int kk=0; kk<dof; kk++){
                    Cx[start + neighbour_idx*dof + kk] += weight * vals[k*dof + kk] / total_weight / total_weight_target;
                }
            }
        }
    }
}

template<typename T> __global__
void SOR_l0(T* K_flat, int* elements_flat, int* el_ids, T* weights, int* sorter, int* node_ids, int n_nodes, T* vec, T* b, T* inv_diagonal, int dof, int elements_size, int elem_flat_size, bool* con_map, T omega) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // spread to every other node for better Gauses-Seidel like behavior
    // int i = (thread*2+(n_nodes+1)%2)%(n_nodes+(n_nodes+1)%2);

    if(i>=n_nodes) return;
    
    int st, en;
    if (i<n_nodes-1) {
        st = node_ids[i];
        en = node_ids[i+1];
    } else {
        st = node_ids[i];
        en = elem_flat_size;
    }

    T local_matvec[3] = {0.0};
    T local_M[9] = {0.0};

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
            local_matvec[k] += val * weight;
            // vec[i*dof+k] += omega * (b[i*dof+k] - val * weight) * inv_diagonal[i*dof+k];
        }
    }

    for (int k=0; k<dof; k++) {
        if (con_map[i*dof+k] == true) {
            continue;
        }
        vec[i*dof+k] += omega * (b[i*dof+k] - local_matvec[k]) * inv_diagonal[i*dof+k];
    }
}

template<typename T> __global__
void SOR_l1p(T* Ax, int* Ap, int* Aj, T* vec, T* b, T* inv_diagonal, int dof, T omega, int n_nodes) {
        
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // spread to every other node for better Gauses-Seidel like behavior
    // int i = (thread*2+(n_nodes+1)%2)%(n_nodes+(n_nodes+1)%2);

    if(i>=n_nodes) return;

    for (int k=0; k<dof; k++) {
        T val = 0.0;
        for (int j=Ap[i*dof + k]; j<Ap[i*dof + k + 1]; j++) {
            val += Ax[j] * vec[Aj[j]];
        }
        vec[i*dof+k] += omega * (b[i*dof+k] - val) * inv_diagonal[i*dof+k];
    }
}

template<typename T> __global__
void csr_matvec(const T* __restrict__ Ax, const int* __restrict__ Ap, const int* __restrict__ Aj, const T* __restrict__ x, T* out, const int n_rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i>=n_rows) return;

    T val=0.0f;
    for (int j=81*i; j<81*i+81; j++) {
        if(Ax[j] == 0.0f) continue;
        val += Ax[j] * x[Aj[j]];
    }
    out[i] = val;
}

template<typename T> __global__
void csr_diagonal(const T* __restrict__ Ax, const int*  Ap, const int* __restrict__ Aj, T* out, const int n_rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i>=n_rows) return;

    for (int j=Ap[i]; j<Ap[i+1]; j++) {
        if (Aj[j] == i) {
            out[i] += Ax[j];
        }
    }
}