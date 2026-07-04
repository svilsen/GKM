neighbourhood_kernel_reconstruction <- function(neighbourhood, kernel, k) {
    ##
    neighbourhood_matrix <- matrix(0, nrow = k, ncol = k)
    kernel_matrix <- matrix(NA, nrow = k, ncol = k)

    ##
    for (i in seq_len(k)) {
        ##
        neighbourhood_i <- neighbourhood[[i]] + 1
        kernel_i <- kernel[[i]]

        ##
        for (j in seq_along(neighbourhood_i)) {
            neighbourhood_matrix[i, neighbourhood_i[j]] <- 1
            kernel_matrix[i, neighbourhood_i[j]] <- kernel_i[j]
        }

    }

    return(list(neighbourhood = neighbourhood_matrix, kernel = kernel_matrix))
}

#' @title Control arguments for the \link{greedy_cluster} function
#'
#' @description Creates a list of default control arguments used by the \link{greedysom} function.
#'
#' @param nd Integer: The maximum distance, measured along the grid, for two clusters to be considered neighbours.
#' @param rngstart Boolean: Should the clusters be initialised entirely at random, or using kmeans++?
#' @param greedy Boolean: Should the assignment be made greedily using the neighbourhood structure?
#' @param greedy_reset Integer: The number of iterations between non-greedy allocations (only used if \code{greedy == TRUE}).
#' @param greedy_end Boolean: Should the final allocation be made greedily?
#' @param competitive Boolean: Should the clusters be allowed to compete during the training process?
#' @param competitive_release Integer: The number of iterations before competitive learning is stopped (only used if \code{competitive == TRUE}).
#' @param batch Boolean: Should the batch version of the algorithm be used?
#' @param max_iteration Numeric vector: A tuple (the maximum number of iterations, the maximum number of iterations below the tolerance).
#' @param learning_rate Numeric vector: A triple (start learning rate, end learning rate, number iterations to go from start to end rate).
#' @param learning_range Numeric vector: A triple (start learning range, end learning range, number iterations to go from start to end range).
#' @param tolerance Numeric: A tolerance on the absolute change in centres between iterations; used for early stopping.
#' @param seed Numeric: A seed passed to c++.
#' @param trace Integer: Prints tracing information on the progress of the algorithm every \code{trace} number of iterations.
#'
#' @details The argument \code{competitive_release} is technically not necessary as it can be controlled by setting an extremely small learning range. However, when the learning range is low (to the point where the optimisation is not competitive) the performance computational complexity of the algorithm can be improved by not accounting for neighbours.
#'
#' @return A list of control arguments.
#' @export
control_greedy_cluster <- function(
        nd = 2L,
        rngstart = FALSE,
        greedy = TRUE,
        greedy_reset = 10,
        greedy_end = FALSE,
        competitive = FALSE,
        competitive_release = NULL,
        max_iteration = c(100, 10),
        learning_range = c(1.0, 0.1, 5),
        seed = sample(1e6, 1),
        tolerance = (.Machine$double.eps)^(1/4),
        trace = 0L
) {
    ##
    if (!is.infinite(nd)) {
        if (!is.numeric(nd)) {
            stop("'nd' has to be an integer (or at least numeric).")
        }
        else {
            nd <- as.integer(nd)
        }
    }

    ##
    if (!is.logical(rngstart)) {
        stop("'rngstart' has to be logical.")
    }

    ##
    if (!is.logical(greedy)) {
        stop("'greedy' has to be logical.")
    }

    ##
    if (!is.logical(competitive)) {
        stop("'competitive' has to be logical.")
    }

    ##
    if (!is.vector(max_iteration)) {
        max_iteration <- as.vector(max_iteration)
    }

    if (!is.numeric(max_iteration)) {
        stop("'max_iteration' has to be a vector of numeric values.")
    }

    if (length(max_iteration) != 2) {
        stop("'max_iteration' must contain two elements.")
    }

    ##
    if (is.null(greedy_reset)) {
        greedy_reset <- max_iteration[1] + 1
    }

    if (!is.numeric(greedy_reset)) {
        stop("'greedy_reset' has to be numeric.")
    }

    ##
    if (!is.vector(learning_range)) {
        learning_range <- as.vector(learning_range)
    }

    if (!is.numeric(learning_range)) {
        stop("'learning_range' has to be a vector of numeric values.")
    }

    if (length(learning_range) != 3) {
        stop("'learning_range' has to contain three elements.")
    }

    ##
    if (is.null(competitive_release)) {
        competitive_release <- 2L * learning_range[3]
    }

    if (!is.numeric(competitive_release)) {
        stop("'competitive_release' has to be numeric.")
    }

    ##
    if (!is.numeric(tolerance)) {
        stop("'tolerance' has to be numeric.")
    }

    ##
    if (!is.infinite(trace)) {
        if (!is.numeric(trace)) {
            stop("'trace' has to be an integer (or at least numeric).")
        }
        else {
            trace <- as.integer(trace)
        }
    }

    ##
    return(
        list(
            nd = nd, rngstart = rngstart,
            greedy = greedy, greedy_reset = greedy_reset, greedy_end = greedy_end,
            competitive = competitive, competitive_release = competitive_release,
            max_iteration = max_iteration, learning_range = learning_range,
            tolerance = tolerance, seed = seed, trace = trace
        )
    )
}

#' @title Greedy K-means Clustering
#'
#' @description Regularised K-means clustering, allowing for greedy re-allocation of data-points when updating clusters, by exploiting the pre-defined neighbourhood structure.
#'
#' @param x Numeric matrix: A matrix with rows and columns corresponding to observations and features, respectively. NB: if \code{x} is not a matrix, the function will try to cast it as a matrix.
#' @param k Integer: The number of clusters.
#' @param control List: A list of control arguments, for more details see \link{control_greedy_cluster}.
#'
#' @return An object of class \link{cluster}.
#' @export
greedy_cluster <- function(x, k, control = list()) {
    ##
    #
    control <- do.call(control_greedy_cluster, control)

    #
    if (!is.matrix(x)) {
        x <- as.matrix(x)
    }

    if (!is.numeric(x)) {
        stop("'x' has to be a matrix of numeric values.")
    }

    if (any(is.na(x)) | any(is.nan(x)) | any(is.infinite(x))) {
        stop("'x' contains values which are 'NA', 'NaN', or 'Inf'.")
    }

    #
    if (!is.integer(k)) {
        k <- as.integer(k)
    }

    ##
    #
    res <- greedy_cluster_cpp(
        x = x,
        k = k,
        nd = control[["nd"]],
        rngstart = control[["rngstart"]],
        greedy = control[["greedy"]],
        greedyreset = control[["greedy_reset"]],
        greedyend = control[["greedy_end"]],
        competitive = control[["competitive"]],
        competitiverelease = control[["competitive_release"]],
        lrange = control[["learning_range"]],
        tol = control[["tolerance"]],
        maxiter = control[["max_iteration"]],
        seed = control[["seed"]],
        trace = control[["trace"]]
    )

    # Change from sparse to complete matrix
    res[c("neighbourhood", "kernel")] <- neighbourhood_kernel_reconstruction(
        neighbourhood = res[["neighbourhood"]],
        kernel = res[["kernel"]],
        k = res[["k"]]
    )

    # Change from C++ to R indices.
    res[["previous"]] <- res[["previous"]] + 1

    #
    if (control$data) {
        res["data"] <- x
    }

    ##
    #
    class(res) <- "cluster"

    #
    return(res)
}


#' @rdname predict
#' @method predict cluster
#' @export
predict.cluster <- function(object, ...) {
    ##
    #
    dots <- list(...)

    #
    if (is.null(dots[["newdata"]])) {
        xnew <- object$data
    }
    else {
        xnew <- dots[["newdata"]]
    }

    #
    if (is.null(dots[["reasign"]])) {
        if (is.null(dots[["newdata"]])) {
            reassign <- FALSE
        }
        else {
            reassign <- TRUE
        }
    }
    else {
        reassign <- dots[["reassign"]]
    }

    if (!is.logical(reassign)) {
        reassign <- as.logical(reassign)
    }

    ##
    #
    if (reassign) {
        pred <- predict_cluster_cpp(xnew, object$centres, object$minmax, object$k)
    }
    else {
        pred <- object$previous
    }

    #
    return(pred)
}

