#ifndef create_centres
#define create_centres

#include <RcppArmadillo.h>

//
arma::mat create_random_centres(const arma::mat & x, const int & k, const int & d, const int & seed);
arma::mat create_plusplus_centres(const arma::mat & x, const int & k, const int & d, const bool & competitive, const int & seed);

#endif
