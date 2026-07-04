#include <RcppArmadillo.h>
#include <random>

#include "metrics.hpp"
#include "match.hpp"

//
void bestmatch(int & _idx, double & _d_idx, const arma::vec & v, const arma::mat & centres, const int & d, const int & k) {
    //
    _idx = 0;
    arma::vec centres_idx = centres.row(0).as_col();
    _d_idx = euclidian_distance(v, centres_idx, d);

    //
    for (int i = 1; i < k; i++) {
        centres_idx = centres.row(i).as_col();
        double d_idx_i = euclidian_distance(v, centres_idx, d);
        if (d_idx_i < _d_idx) {
            _idx = i;
            _d_idx = d_idx_i;
        }
    }
}

void greedymatch(int & _idx, double & _d_idx, const int & _p, const arma::vec & v, const arma::mat & centres, const std::vector<std::vector<int>> & neighbourhood, const int & d, const int & k) {
    //
    if (_p < 0) {
        //
        std::random_device dev;
        std::mt19937 rng(dev());

        //
        std::uniform_int_distribution<int> rand_idx(0, k - 1);

        //
        _idx = rand_idx(rng);
    }
    else {
        _idx = _p;
    }

    //
    arma::vec centres_idx = centres.row(_idx).as_col();
    _d_idx = euclidian_distance(v, centres_idx, d);

    std::vector<double> d_complete(k, HUGE_VAL);
    d_complete[_idx] = _d_idx;

    //
    bool new_centre_found = true;
    while (new_centre_found) {
        //
        new_centre_found = false;

        //
        const std::vector<int> & neighbourhood_idx = neighbourhood[_idx];
        const int s = neighbourhood_idx.size();

        //
        for (int i = 0; i < s; i++) {
            double d_idx_i;
            if (d_complete[neighbourhood_idx[i]] < HUGE_VAL) {
                d_idx_i = d_complete[neighbourhood_idx[i]];
            }
            else {
                centres_idx = centres.row(neighbourhood_idx[i]).as_col();
                d_idx_i = euclidian_distance(v, centres_idx, d);
            }

            if (d_idx_i < _d_idx) {
                //
                new_centre_found = true;

                //
                _idx = neighbourhood_idx[i];
                _d_idx = d_idx_i;
            }
        }
    }
}

