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
    int nd;

    //
    bool rngstart;

    //
    bool greedy;
    int greedy_reset;
    bool greedy_end;

    bool competitive;
    int competitive_release;

    //
    arma::vec lrange;
    int iterations;

    //
    std::vector<std::vector<double>> kernel;
    std::vector<std::vector<int>> neighbourhood;

    //
    arma::mat centres;

    //
    std::vector<int> previous;
    arma::mat minmax;

    //// Functions
    // Constructor(s)
    GreedyCluster(
        const int & _d, const int & _k, const int & _nd, const bool & _rngstart,
        const bool & _greedy, const int & _greedy_reset, const bool & _greedy_end,
        const bool & _competitive, const int & _competitive_release,
        const arma::colvec & _lrange
    );

    //// Initialisers
    // Neighbourhood, kernel, and index-order
    void initialise_neighbourhood();

    // Clusters
    void find_minmax(arma::mat & _minmax, const arma::mat & x);
    void standardise_data(arma::mat & _z, const arma::mat & x);
    arma::mat unstandardise_centres();

    void initialise_clusters(const arma::mat & x, const int & seed);

    //// Update clusters
    // Update learning rate and range
    double update_lrange(const int & n);

    // Clusters
    void update_centres_batch(arma::mat & _kernel_val_sum, arma::vec & _kernel_sum, const arma::vec & v, const int & idx, const int & n);
    void update_clusters_batch(const arma::mat & x, const double & tol, const arma::vec & maxiter, const int & trace);
};

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
);

std::vector<int> predict_cluster_cpp(
        const arma::mat & x,
        const arma::mat & centres
);

#endif
