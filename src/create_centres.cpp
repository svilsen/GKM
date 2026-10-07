#include <RcppArmadillo.h>

#include <random>
#include <unordered_set>

#include "create_centres.hpp"
#include "metrics.hpp"

// Completely random centre allocation
arma::mat create_random_centres(const arma::mat & x, const int & k, const int & d, const int & seed) {
    //
    const int & M = x.n_rows;

    //
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> random_index(0, M - 1);
    std::unordered_set<int> used_elements;

    //
    arma::mat _centres(k, d);
    for (int i = 0; i < k; i++) {
        double idx_i = random_index(rng);
        while (used_elements.find(idx_i) != used_elements.end()) {
            idx_i = random_index(rng);
        }

        used_elements.insert(idx_i);
        _centres.row(i) = x.row(idx_i);
    }

    return _centres;
}

// K-means++ style centre allocation
void update_x_distance(arma::vec & accumulated_distance, const arma::vec & centre, const arma::mat & x, const int & d) {
    //
    const int & M = x.n_rows;
    for (int m = 0; m < M; m++) {
        const arma::vec & x_m = x.row(m).as_col();

        // Could technically be any power 'p', would make it an Lp formulation of the same problem.
        const double new_dist_m = std::pow(euclidian_distance(x_m, centre, d), 2.0);
        double dist_m = accumulated_distance[m];
        if (m > 0) {
            dist_m = dist_m - accumulated_distance[m - 1];
        }

        if (new_dist_m < dist_m) {
            accumulated_distance[m] = new_dist_m;
            if (m > 0) {
                accumulated_distance[m] = accumulated_distance[m] + accumulated_distance[m - 1];
            }
        }
    }
}

int update_index(const double & u, const arma::vec & accumulated_distance) {
    //
    const int & M = accumulated_distance.size();
    const double & reciprocal_sum_distance = 1.0 / accumulated_distance[M - 1];

    //
    int m = 0;
    double distance_m = accumulated_distance[m];
    double ratio_m = reciprocal_sum_distance * distance_m;

    //
    bool continue_search = u > ratio_m;
    while (continue_search) {
        //
        m++;
        const double & accumulated_distance_m = accumulated_distance[m];

        //
        distance_m = accumulated_distance_m - distance_m;
        ratio_m = reciprocal_sum_distance * accumulated_distance_m;

        //
        continue_search = ((u > ratio_m) | (distance_m == 0)) & (m < M);
    }

    return m;
}


//
arma::mat create_plusplus_centres(const arma::mat & x, const int & k, const int & d, const bool & competitive, const int & seed) {
    //
    const int & M = x.n_rows;
    arma::mat _centres(k, d);

    //
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> random_01(0, 1);

    //
    const int idx = static_cast<int>(M * random_01(rng));
    _centres.row(0) = x.row(idx);

    // NB: DONE MANUALLY DUE TO PROBLEMS WHEN FILLING 'accumulated_distance' USING HUGE_VAL WITH ARMA INTERFACE
    arma::vec accumulated_distance(M);
    for (int i = 0; i < M; i++) {
        accumulated_distance[i] = HUGE_VAL;
    }

    //
    update_x_distance(accumulated_distance, _centres.row(0).as_col(), x, d);

    //
    for (int i = 1; i < k; i++) {
        //
        const double u = random_01(rng);
        const int idx_i = update_index(u, accumulated_distance);

        //
        _centres.row(i) = x.row(idx_i);

        //
        update_x_distance(accumulated_distance, _centres.row(i).as_col(), x, d);
    }

    //
    return _centres;
}

