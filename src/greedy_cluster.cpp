#include <RcppArmadillo.h>
#include <random>
#include <unordered_set>

#include "match.hpp"
#include "metrics.hpp"
#include "create_centres.hpp"
#include "greedy_cluster.hpp"

//// Constructors
GreedyCluster::GreedyCluster(
    const int & _d,
    const int & _k,
    const int & _nd,
    const bool & _rngstart,
    const bool & _greedy,
    const int & _greedy_reset,
    const bool & _greedy_end,
    const bool & _competitive,
    const int & _competitive_release,
    const arma::vec & _lrange
) : d(_d), k(_k), nd(_nd), rngstart(_rngstart), greedy(_greedy), greedy_reset(_greedy_reset), greedy_end(_greedy_end), competitive(_competitive), competitive_release(_competitive_release), lrange(_lrange) {
    //
    neighbourhood = std::vector<std::vector<int>>(k);
    kernel = std::vector<std::vector<double>>(k);
}

//// Initialisation
// Neighbourhood and kernel
void GreedyCluster::initialise_neighbourhood() {
    //
    arma::mat distances = arma::zeros(k, k);
    for (int i = 0; i < k; i++) {
        //
        if (i < k - 1) {
            for (int j = i + 1; j < k; j++) {
                if (j != i) {
                    double distance_ij = euclidian_distance(centres.row(i).as_col(), centres.row(j).as_col(), d);
                    distances(i, j) = distance_ij;
                    distances(j, i) = distance_ij;
                }
            }
        }

        //
        arma::uvec sorted_distance_i = arma::sort_index(distances.row(i));

        //
        std::vector<int> neighbourhood_i(nd, 0);
        std::vector<double> kernel_i(nd, HUGE_VAL);
        for (int j = 0; j < nd; j++) {
            neighbourhood_i[j] = sorted_distance_i[j + 1];
            kernel_i[j] = 1.0;
        }

        //
        neighbourhood[i] = neighbourhood_i;
        kernel[i] = kernel_i;
    }

}

// Clusters
void GreedyCluster::find_minmax(arma::mat & _minmax, const arma::mat & x) {
    //
    const int & M = x.n_rows;

    //
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < d; j++) {
            const double & x_ij = x(i, j);
            if (x_ij < _minmax(0, j)) {
                _minmax(0, j) = x_ij;
            }

            if (x_ij > _minmax(1, j)) {
                _minmax(1, j) = x_ij;
            }
        }
    }
}

void GreedyCluster::standardise_data(arma::mat & _z, const arma::mat & x) {
    //
    const int & M = x.n_rows;

    //
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < d; j++) {
            _z(i, j) = (x(i, j) - minmax(0, j)) / (minmax(1, j) - minmax(0, j));
        }
    }
}

arma::mat GreedyCluster::unstandardise_centres() {
    //
    arma::mat _centres(k, d);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < d; j++) {
            _centres(i, j) = centres(i, j) * (minmax(1, j) - minmax(0, j)) + minmax(0, j);
        }
    }

    return _centres;
}

void GreedyCluster::initialise_clusters(const arma::mat & x, const int & seed) {
    ////
    //
    minmax = arma::zeros(2, d);
    find_minmax(minmax, x);

    ////
    //
    if (rngstart) {
        centres = create_random_centres(x, k, d, seed);
    }
    else {
        centres = create_plusplus_centres(x, k, d, competitive, seed);
    }

}

//// Update
// Learning-range
double GreedyCluster::update_lrange(const int & n) {
    double lrng = lrange[1];
    if (n < lrange[2]) {
        lrng = lrange[0] + (lrange[1] - lrange[0]) * n / lrange[2];
    }

    return lrng;
}

//// Clusters
// Batch
void GreedyCluster::update_centres_batch(arma::mat & _kernel_val_sum, arma::vec & _kernel_sum, const arma::vec & v, const int & idx, const int & n) {
    //
    const std::vector<int> & neighbourhood_idx = neighbourhood[idx];
    const int s = neighbourhood_idx.size();

    //
    const std::vector<double> & kernel_idx = kernel[idx];
    const double & lrng = update_lrange(n);

    //
    for (int j = 0; j < d; j++) {
        // Update centre of idx
        _kernel_val_sum(idx, j) += v[j];
        if (j == 0) {
            _kernel_sum(idx) += 1.0;
        }

        // Update centres for neighbours of idx
        if (competitive & (n < competitive_release)) {
            for (int i = 0; i < s; i++) {
                //
                const double kernel_i = std::exp(-(kernel_idx[i] * kernel_idx[i]) / (2.0 * lrng));
                const double kernel_val_ij = kernel_i * v[j];

                //
                _kernel_val_sum(neighbourhood_idx[i], j) += kernel_val_ij;

                if (j == 0) {
                    _kernel_sum(neighbourhood_idx[i]) += kernel_i;
                }
            }
        }
    }
}

void GreedyCluster::update_clusters_batch(const arma::mat & x, const double & tolerance, const arma::vec & maxiter, const int & trace) {
    //
    const int & M = x.n_rows;

    //
    arma::mat z(M, d);
    standardise_data(z, x);

    //
    previous = std::vector<int>(M);
    std::fill(previous.begin(), previous.end(), -1);

    //
    int n = 0;
    int n_delta_centres = 0;

    //
    bool not_stopping = n < maxiter[0];
    while (not_stopping) {
        //
        arma::mat kernel_val_sum = arma::zeros(k, d);
        arma::vec kernel_sum = arma::zeros(k);
        for (int m = 0; m < M; m++) {
            // MIGHT BE EASIER TO CHANGE EVERYTHING TO USE ROW VECTORS
            const arma::vec & v = z.row(m).as_col();

            //
            int idx_n;
            double d_idx_n;
            if (greedy & (n > 0) & ((n % greedy_reset) != 0)) {
                greedymatch(idx_n, d_idx_n, previous[m], v, centres, neighbourhood, d, k);
            }
            else {
                bestmatch(idx_n, d_idx_n, v, centres, d, k);
            }

            previous[m] = idx_n;
            update_centres_batch(kernel_val_sum, kernel_sum, v, idx_n, n);
        }

        //
        double delta_centres = 0;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < d; j++) {
                const double centres_ij = centres(i, j);
                if (kernel_sum[i] > 1e-8) {
                    centres(i, j) = kernel_val_sum(i, j) / kernel_sum[i];
                }

                delta_centres += std::abs(centres(i, j) - centres_ij);
            }
        }

        //
        n++;

        //
        if (n >= maxiter[0]) {
            not_stopping = false;
        }
        else {
            if (delta_centres < tolerance) {
                n_delta_centres++;
                if (n_delta_centres >= maxiter[1]) {
                    not_stopping = false;
                }
            }
            else {
                n_delta_centres = 0;
            }
        }

        //
        if (trace > 0) {
            if ((n == 1) | (n == maxiter[0]) | (n % trace == 0) | (!not_stopping)) {
                Rcpp::Rcout << "Iteration: " << n << " / " << maxiter[0] << " :: Change in centre(s): " << delta_centres << " :: Below threshold: " << n_delta_centres << " / " << maxiter[1] << "\n";
            }
        }
    }

    //
    for (int m = 0; m < M; m++) {
        const arma::vec & v = z.row(m).as_col();

        //
        int idx_n;
        double d_idx_n;
        if (greedy_end) {
            greedymatch(idx_n, d_idx_n, previous[m], v, centres, neighbourhood, d, k);
        }
        else {
            bestmatch(idx_n, d_idx_n, v, centres, d, k);
        }

        previous[m] = idx_n;
    }

    //
    iterations = n;
}


//// R INTERFACE / EXPORT FUNCTIONS
//[[Rcpp::export()]]
Rcpp::List greedy_cluster_cpp(
        const arma::mat & x,
        const int & k,
        const int & nd,
        const bool & rngstart,
        const bool & greedy,
        const int & greedy_reset,
        const bool & greedy_end,
        const bool & competitive,
        const int & competitive_release,
        const arma::vec & lrange,
        const double & tol,
        const arma::vec & maxiter,
        const int & seed,
        const int & trace
) {
    //
    const int & d = x.n_cols;

    //
    GreedyCluster GRCL(d, k, nd, rngstart, greedy, greedy_reset, greedy_end, competitive, competitive_release, lrange);

    GRCL.initialise_clusters(x, seed);
    GRCL.initialise_neighbourhood();

    //
    GRCL.update_clusters_batch(x, tol, maxiter, trace);

    //
    arma::mat ucentres = GRCL.unstandardise_centres();

    //
    return Rcpp::List::create(
        Rcpp::Named("k") = GRCL.k,
        Rcpp::Named("centres") = ucentres,
        Rcpp::Named("neighbourhood") = GRCL.neighbourhood,
        Rcpp::Named("kernel") = GRCL.kernel,
        Rcpp::Named("previous") = GRCL.previous,
        Rcpp::Named("iterations") = GRCL.iterations,
        Rcpp::Named("minmax") = GRCL.minmax
    );
}

//[[Rcpp::export()]]
std::vector<int> predict_cluster_cpp(
    const arma::mat & x,
    const arma::mat & centres,
    const arma::mat & minmax,
    const int & k
) {
    //
    const int M = x.n_rows;
    const int d = x.n_cols;

    //
    std::vector<int> pred(M);
    for (int m = 0; m < M; m++) {
        //
        arma::vec v_m = x.row(m).as_col();
        for (int j = 0; j < d; j++) {
            v_m[j] = (x[j] - minmax(0, j)) / (minmax(1, j) - minmax(0, j));
        }

        //
        int idx_m;
        double d_idx_m;
        bestmatch(idx_m, d_idx_m, v_m, centres, k, d);

        //
        pred[m] = idx_m;
    }

    //
    return pred;
}

