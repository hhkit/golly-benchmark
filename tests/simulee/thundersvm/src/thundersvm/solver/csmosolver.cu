//
// Created by jiashuai on 17-10-25.
//
#include <thundersvm/solver/csmosolver.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/detail/par.h>
#include <thundersvm/kernel/smo_kernel.h>

void CSMOSolver::solve(const KernelMatrix &k_mat, const SyncData<int> &y, SyncData<real> &alpha, real &rho,
                       SyncData<real> &f_val, real eps, real C, int ws_size) const {
    uint n_instances = k_mat.n_instances();
    uint q = ws_size / 2;

    SyncData<int> working_set(ws_size);
    SyncData<int> working_set_first_half(q);
    SyncData<int> working_set_last_half(q);
    working_set_first_half.set_device_data(working_set.device_data());
    working_set_last_half.set_device_data(&working_set.device_data()[q]);
    working_set_first_half.set_host_data(working_set.host_data());
    working_set_last_half.set_host_data(&working_set.host_data()[q]);

    SyncData<int> f_idx(n_instances);
    SyncData<int> f_idx2sort(n_instances);
    SyncData<real> f_val2sort(n_instances);
    SyncData<real> alpha_diff(ws_size);
    SyncData<real> diff_and_bias(2);

    SyncData<real> k_mat_rows(ws_size * k_mat.n_instances());
    SyncData<real> k_mat_rows_first_half(q * k_mat.n_instances());
    SyncData<real> k_mat_rows_last_half(q * k_mat.n_instances());
    k_mat_rows_first_half.set_device_data(k_mat_rows.device_data());
    k_mat_rows_last_half.set_device_data(&k_mat_rows.device_data()[q * k_mat.n_instances()]);
    for (int i = 0; i < n_instances; ++i) {
        f_idx[i] = i;
    }
    init_f(alpha, y, k_mat, f_val);
    LOG(INFO) << "training start";
    for (int iter = 1;; ++iter) {
        //select working set
        f_idx2sort.copy_from(f_idx);
        f_val2sort.copy_from(f_val);
        thrust::sort_by_key(thrust::cuda::par, f_val2sort.device_data(), f_val2sort.device_data() + n_instances,
                            f_idx2sort.device_data(), thrust::less<real>());
        vector<int> ws_indicator(n_instances, 0);
        if (1 == iter) {
            select_working_set(ws_indicator, f_idx2sort, y, alpha, C, working_set);
            k_mat.get_rows(working_set, k_mat_rows);
        } else {
            working_set_first_half.copy_from(working_set_last_half);
            for (int i = 0; i < q; ++i) {
                ws_indicator[working_set[i]] = 1;
            }
            select_working_set(ws_indicator, f_idx2sort, y, alpha, C, working_set_last_half);
            k_mat_rows_first_half.copy_from(k_mat_rows_last_half);
            k_mat.get_rows(working_set_last_half, k_mat_rows_last_half);
        }
        //local smo
        smo_kernel(y.device_data(), f_val.device_data(), alpha.device_data(), alpha_diff.device_data(),
                   working_set.device_data(), ws_size, C, k_mat_rows.device_data(), k_mat.diag().device_data(),
                   n_instances, eps, diff_and_bias.device_data());
        //update f
        SAFE_KERNEL_LAUNCH(update_f, f_val.device_data(), ws_size, alpha_diff.device_data(), k_mat_rows.device_data(),
                           n_instances);
        LOG_EVERY_N(10, INFO) << "diff=" << diff_and_bias[0];
        if (diff_and_bias[0] < eps) {
            rho = calculate_rho(f_val, y, alpha, C);
            break;
        }
    }
}

void CSMOSolver::select_working_set(vector<int> &ws_indicator, const SyncData<int> &f_idx2sort, const SyncData<int> &y,
                                    const SyncData<real> &alpha, real C, SyncData<int> &working_set) const {
    int n_instances = ws_indicator.size();
    int p_left = 0;
    int p_right = n_instances - 1;
    int n_selected = 0;
    const int *index = f_idx2sort.host_data();
    while (n_selected < working_set.size()) {
        int i;
        if (p_left < n_instances) {
            i = index[p_left];
            while (ws_indicator[i] == 1 || !is_I_up(alpha[i], y[i], C)) {
                //construct working set of I_up
                p_left++;
                if (p_left == n_instances) break;
                i = index[p_left];
            }
            if (p_left < n_instances) {
                working_set[n_selected++] = i;
                ws_indicator[i] = 1;
            }
        }
        if (p_right >= 0) {
            i = index[p_right];
            while (ws_indicator[i] == 1 || !is_I_low(alpha[i], y[i], C)) {
                //construct working set of I_low
                p_right--;
                if (p_right == -1) break;
                i = index[p_right];
            }
            if (p_right >= 0) {
                working_set[n_selected++] = i;
                ws_indicator[i] = 1;
            }
        }

    }
}

real
CSMOSolver::calculate_rho(const SyncData<real> &f_val, const SyncData<int> &y, SyncData<real> &alpha, real C) const {
    int n_free = 0;
    real sum_free = 0;
    real up_value = INFINITY;
    real low_value = -INFINITY;
    for (int i = 0; i < alpha.size(); ++i) {
        if (alpha[i] > 0 && alpha[i] < C) {
            n_free++;
            sum_free += f_val[i];
        }
        if (is_I_up(alpha[i], y[i], C)) up_value = min(up_value, f_val[i]);
        if (is_I_low(alpha[i], y[i], C)) low_value = max(low_value, f_val[i]);
    }
    return 0 != n_free ? (sum_free / n_free) : (-(up_value + low_value) / 2);
}

void CSMOSolver::init_f(const SyncData<real> &alpha, const SyncData<int> &y, const KernelMatrix &k_mat,
                        SyncData<real> &f_val) const {
    //TODO initialize with smaller batch to reduce memory usage
    vector<int> idx_vec;
    vector<real> alpha_diff_vec;
    for (int i = 0; i < alpha.size(); ++i) {
        if (alpha[i] != 0) {
            idx_vec.push_back(i);
            alpha_diff_vec.push_back(-alpha[i] * y[i]);
        }
    }
    if (idx_vec.size() > 0) {
        SyncData<int> idx(idx_vec.size());
        SyncData<real> alpha_diff(idx_vec.size());
        idx.copy_from(idx_vec.data(), idx_vec.size());
        alpha_diff.copy_from(alpha_diff_vec.data(), idx_vec.size());
        SyncData<real> kernel_rows(idx.size() * k_mat.n_instances());
        k_mat.get_rows(idx, kernel_rows);
        SAFE_KERNEL_LAUNCH(update_f, f_val.device_data(), idx.size(), alpha_diff.device_data(),
                           kernel_rows.device_data(), k_mat.n_instances());
    }
}

void CSMOSolver::smo_kernel(const int *label, real *f_values, real *alpha, real *alpha_diff, const int *working_set,
                            int ws_size, float C, const float *k_mat_rows, const float *k_mat_diag, int row_len,
                            real eps, real *diff_and_bias) const {
    size_t smem_size = ws_size * sizeof(real) * 3 + 2 * sizeof(float);
    c_smo_solve_kernel << < 1, ws_size, smem_size >> >
                                        (label, f_values, alpha, alpha_diff,
                                                working_set, ws_size, C, k_mat_rows,
                                                k_mat_diag, row_len, eps, diff_and_bias);
}
