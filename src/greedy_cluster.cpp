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
    const std::vector<std::vector<int>> & _neighbourhood,
    const bool & _rng_start,
    const int & _greedy,
    const bool & _greedy_end,
    const bool & _competitive,
    const int & _competitive_release,
    const arma::vec & _lrange
) : d(_d), k(_k), neighbourhood(_neighbourhood), rng_start(_rng_start), greedy(_greedy), greedy_end(_greedy_end), competitive(_competitive), competitive_release(_competitive_release), lrange(_lrange) { }

//// Initialisation
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

//
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

//
void GreedyCluster::unstandardise_centres(arma::mat & _centres) {
    //
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < d; j++) {
            _centres(i, j) = _centres(i, j) * (minmax(1, j) - minmax(0, j)) + minmax(0, j);
        }
    }
}

//
void GreedyCluster::reorder_clusters(arma::mat & _centres) {
    //
    arma::mat centres_ordered = arma::mat(k, d);
    std::vector<bool> centres_assigned(k, false);
    std::vector<bool> neighbour_assigned(k, false);

    //
    int idx = 0;
    int degree_max = neighbourhood[0].size();
    for (int i = 1; i < k; i++) {
        if (neighbourhood[i].size() > degree_max) {
            idx = i;
            degree_max = neighbourhood[i].size();
        }
    }

    centres_ordered.row(idx) = _centres.row(0);
    centres_assigned[0] = true;
    neighbour_assigned[idx] = true;

    //
    for (int h = 1; h < k; h++) {
        //
        double score_min = HUGE_VAL;
        int idx_best = -1;
        int neighbour_best = -1;

        //
        for (int i = 0; i < k; i++) {
            if (neighbour_assigned[i]) {
                continue;
            }

            for (int j = 0; j < k; j++) {
                if (centres_assigned[j]) {
                    continue;
                }

                double score = 0.0;
                for (int n : neighbourhood[j]) {
                    if (neighbour_assigned[n]) {
                        score += euclidian_distance(_centres.row(j).as_col(), centres_ordered.row(n).as_col(), d);
                    }
                }

                if (score < score_min) {
                    score_min = score;
                    idx_best = j;
                    neighbour_best = i;
                }
            }
        }

        if ((neighbour_best > 0) && (idx_best > 0)) {
            centres_ordered.row(neighbour_best) = _centres.row(idx_best);
            centres_assigned[idx_best] = true;
            neighbour_assigned[neighbour_best] = true;
        }
    }

    _centres = centres_ordered;
}

//
void GreedyCluster::initialise_clusters(arma::mat & z, const arma::mat & x, const int & seed, const int & maxiter) {
    ////
    //
    minmax = arma::zeros(2, d);
    find_minmax(minmax, x);

    //
    standardise_data(z, x);

    ////
    //
    if (rng_start) {
        centres = create_random_centres(z, k, d, seed);
    }
    else {
        centres = create_plusplus_centres(z, k, d, competitive, seed);
    }
}

//// Update
// Learning-range
double GreedyCluster::update_lrange(const int & n, const int & s) {
    double lrng = lrange[1];
    if (n < lrange[2]) {
        lrng = lrange[0] + (lrange[1] - lrange[0]) * n / lrange[2];
    }

    return lrng / s;
}

// Centres
void GreedyCluster::update_centres(arma::mat & _kernel_val_sum, arma::vec & _kernel_sum, const arma::vec & v, const int & idx, const int & n) {
    //
    const std::vector<int> & neighbourhood_idx = neighbourhood[idx];
    const int s = neighbourhood_idx.size();

    //
    const double & lrng = update_lrange(n, s);

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
                const double kernel_i = std::exp(-1.0 / lrng);
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

// Clusters
void GreedyCluster::update_clusters(const arma::mat & z, const double & tolerance, const arma::vec & maxiter, const int & trace) {
    //
    const int & M = z.n_rows;

    //
    previous = std::vector<int>(M);
    std::fill(previous.begin(), previous.end(), -1);

    //
    int n = 0;
    int b = -1;
    int n_delta_centres = 0;

    //
    bool not_stopping = true;
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
            if ((n % greedy) == 0) {
                bestmatch(idx_n, d_idx_n, v, centres, d, k);
                b++;
            }
            else {
                greedymatch(idx_n, d_idx_n, previous[m], v, centres, neighbourhood, d, k);
            }

            previous[m] = idx_n;
            update_centres(kernel_val_sum, kernel_sum, v, idx_n, b);
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
        const std::vector<std::vector<int>> & neighbourhood,
        const bool & rng_start,
        const int & greedy,
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
    const int & n = x.n_rows;

    //
    GreedyCluster GRCL(d, k, neighbourhood, rng_start, greedy, greedy_end, competitive, competitive_release, lrange);

    //
    arma::mat z(n, d);
    GRCL.initialise_clusters(z, x, seed, maxiter[0]);

    //
    GRCL.update_clusters(z, tol, maxiter, trace);

    //
    arma::mat ucentres = GRCL.centres;
    GRCL.unstandardise_centres(ucentres);

    //
    return Rcpp::List::create(
        Rcpp::Named("k") = GRCL.k,
        Rcpp::Named("centres") = ucentres,
        Rcpp::Named("neighbourhood") = GRCL.neighbourhood,
        Rcpp::Named("previous") = GRCL.previous,
        Rcpp::Named("iterations") = GRCL.iterations,
        Rcpp::Named("greedy") = GRCL.greedy,
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

