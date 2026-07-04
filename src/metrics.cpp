#include <RcppArmadillo.h>
#include "metrics.hpp"

//
double euclidian_distance(const arma::vec & v, const arma::vec & w, const int & d) {
    double ed = 0.0;
    for (int i = 0; i < d; i++) {
        const double d_i = (v[i] - w[i]);
        const double sq_d_i = d_i * d_i;
        ed += sq_d_i;
    }
    
    ed = std::sqrt(ed);
    return ed;
}
