#' @keywords internal
"_PACKAGE"

#' @title Greedy K-means Clustering
#'
#' @importFrom Rcpp evalCpp
#'
#' @useDynLib GKM
#'
#' @name GKM-package
#'
#' @rdname GKM-package
NULL

#' @title A cluster object
#'
#' @description A cluster object is a list containing the following:
#' \describe{
#'      \item{\code{"k"}}{The number of clusters.}
#'      \item{\code{"centres"}}{A matrix where each row contains the optimised cluster centres.}
#'      \item{\code{"neighbourhood"}}{A matrix specifying the neighbourhood of the cluster centres.}
#'      \item{\code{"kernel"}}{A matrix specifying the kernel of the cluster centres.}
#'      \item{\code{"previous"}}{A vector indicating the cluster where each data-point was assigned.}
#'      \item{\code{"iterations"}}{The number of iterations before convergence.}
#'      \item{\code{"minmax"}}{A matrix containing the smallest and largest values found along each dimension.}
#' }
#'
#' @name cluster
#' @rdname cluster
NULL
