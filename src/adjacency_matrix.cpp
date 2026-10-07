#include <RcppArmadillo.h>
#include <queue>

////
//
bool is_reachable(const arma::mat & adj_matrix, int start) {
    ////
    //
    int k = adj_matrix.n_rows;

    //
    arma::vec visited(k, false);

    //
    std::queue<int> q;
    q.push(start);

    //
    visited[start] = true;
    int count = 1;
    while (!q.empty()) {
        //
        int & c = q.front();
        q.pop();

        for (int n = 0; n < k; n++) {
            if (adj_matrix(c, n) == 1 && !visited[n]) {
                //
                visited[n] = true;
                q.push(n);

                //
                count++;
            }
        }
    }

    return count == k;
}

//[[Rcpp::export()]]
bool is_connected_matrix_directed(const arma::mat & adj_matrix) {
    ////
    //
    int k = adj_matrix.n_rows;
    for (int i = 0; i < k; ++i) {
        //
        if (!is_reachable(adj_matrix, i)) {
            return false;
        }
    }

    return true;
}

//[[Rcpp::export()]]
std::vector<std::vector<int>> create_adjacency_list_regular(const int & k, const int & neighbours) {
    ////
    //
    int half_neighbours;
    if ((neighbours % 2) == 0) {
        half_neighbours = neighbours / 2;
    }
    else {
        half_neighbours = (neighbours - 1) / 2;
    }

    std::vector<std::vector<int>> adj_list(k);
    for (int i = 0; i < k; i++) {
        std::vector<int> adj_list_i;
        adj_list_i.reserve(neighbours);

        for (int j = 0; j < half_neighbours; j++) {
            const int l = ((i + j + 1) % k);
            const int r = ((i - j - 1 + k) % k);

            adj_list_i.push_back(l);
            adj_list_i.push_back(r);
        }

        if (((neighbours % 2) != 0) & (i < k - (k % 2))) {
            const int l = (i + half_neighbours + 1) % k;
            adj_list_i.push_back(l);
        }

        adj_list[i] = adj_list_i;
    }

    return adj_list;
}

//[[Rcpp::export()]]
std::vector<std::vector<int>> create_adjacency_list_grid(const std::vector<int> & k) {
    //
    const int d = k.size();

    //
    int k_total = 1;
    for (int i = 0; i < d; i++) {
        k_total *= k[i];
    }

    std::vector<int> step_size(d, 1);
    for (int i = 1; i < d; i++) {
        step_size[i] = step_size[i - 1] * k[i - 1];
    }

    //
    std::vector<std::vector<int>> neighbourhood(k_total);
    for (int i = 0; i < k_total; i++) {
        std::vector<int> neighbourhood_i;
        neighbourhood_i.reserve(2 * d);

        for (int j = 0; j < d; ++j) {
            int l = (i / step_size[j]) % k[j];

            if (l > 0) {
                neighbourhood_i.push_back(i - step_size[j]);
            }
            if (l < (k[j] - 1)) {
                neighbourhood_i.push_back(i + step_size[j]);
            }
        }

        //
        neighbourhood[i] = neighbourhood_i;
    }

    return neighbourhood;
}

//[[Rcpp::export()]]
arma::mat neighbourhood_reconstruction(const std::vector<std::vector<int>> & neighbourhood, const int & k) {
    arma::mat neighbourhood_matrix = arma::zeros(k, k);
    for (int i = 0; i < k; ++i) {
        const std::vector<int> & neighbourhood_i = neighbourhood[i];
        for (int j = 0; j < neighbourhood_i.size(); j++) {
            const int & col = neighbourhood_i[j];
            neighbourhood_matrix(i, col) = 1;
        }
    }

    return neighbourhood_matrix;
}

