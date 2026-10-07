#ifndef greedy_cluster
#define greedy_cluster

#include <RcppArmadillo.h>

class GreedyCluster {
public:
    ////
    //
    int d;
    int k;

    //
    int neighbours;

    //
    bool rng_start;

    //
    int greedy;
    bool greedy_end;

    bool competitive;
    int competitive_release;

    //
    arma::vec lrange;
    int iterations;

    //
    std::vector<std::vector<int>> neighbourhood;

    //
    arma::mat centres;

    //
    std::vector<int> previous;
    arma::mat minmax;

    //// Functions
    // Constructor(s)
    GreedyCluster(
        const int & _d,
        const int & _k,
        const std::vector<std::vector<int>> & _neighbourhood,
        const bool & _rng_start,
        const int & _greedy,
        const bool & _greedy_end,
        const bool & _competitive,
        const int & _competitive_release,
        const arma::colvec & _lrange
    );

    //// Initialisers
    // Data
    void find_minmax(arma::mat & _minmax, const arma::mat & x);
    void standardise_data(arma::mat & _z, const arma::mat & x);
    void unstandardise_centres(arma::mat & _centres);

    // Clusters
    void reorder_clusters(arma::mat & _centres);
    void initialise_clusters(arma::mat & z, const arma::mat & x, const int & seed, const int & maxiter);

    //// Update clusters
    // Update learning rate and range
    double update_lrange(const int & n, const int & s);

    // Clusters
    void update_centres(arma::mat & _kernel_val_sum, arma::vec & _kernel_sum, const arma::vec & v, const int & idx, const int & n);
    void update_clusters(const arma::mat & z, const double & tol, const arma::vec & maxiter, const int & trace);
};

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
);

std::vector<int> predict_cluster_cpp(
        const arma::mat & x,
        const arma::mat & centres
);

#endif
