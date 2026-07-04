#ifndef match
#define match

#include <RcppArmadillo.h>

//
void bestmatch(int & _idx, double & _d_idx, const arma::vec & v, const arma::mat & centres, const int & d, const int & k);
void greedymatch(int & _idx, double & _d_idx, const int & _previous, const arma::vec & v, const arma::mat & centres, const std::vector<std::vector<int>> & neighbourhood, const int & d, const int & k);

#endif
