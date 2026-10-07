##
#' @title Control arguments for the \link{greedy_cluster} function
#'
#' @description Creates a list of default control arguments used by the \link{greedysom} function.
#'
#' @param neighbours Integer: The number of neighbours of each cluster.
#' @param neighbours_type String: The type of adjacency matrix constructed from \code{neighbours}.
#' @param neighbours_connected Boolean: Should the neighbourhood graph be connected?
#' @param rng_start Boolean: Should the clusters be initialised entirely at random, or using kmeans++?
#' @param greedy Integer: The number of iterations using best-match allocation before switching to greedy allocation.
#' @param greedy_end Boolean: Should the final allocation be greedy?
#' @param competitive Boolean: Should the clusters be allowed to compete during the training process?
#' @param competitive_release Integer: The number of iterations before competitive learning is stopped (only used if \code{competitive == TRUE}).
#' @param batch Boolean: Should the batch version of the algorithm be used?
#' @param max_iteration Numeric vector: A tuple (the maximum number of iterations, the maximum number of iterations below the tolerance).
#' @param learning_rate Numeric vector: A triple (start learning rate, end learning rate, number iterations to go from start to end rate).
#' @param learning_range Numeric vector: A triple (start learning range, end learning range, number iterations to go from start to end range).
#' @param tolerance Numeric: A tolerance on the absolute change in centres between iterations; used for early stopping.
#' @param seed Numeric: A seed passed to c++.
#' @param include_data Boolean: Should the data be included in the return object?
#' @param trace Integer: Prints tracing information on the progress of the algorithm every \code{trace} number of iterations.
#'
#' @details The argument \code{competitive_release} is technically not necessary as it can be controlled by setting an extremely small learning range. However, when the learning range is low (to the point where the optimisation is not competitive) the performance computational complexity of the algorithm can be improved by not accounting for neighbours.
#'
#' @return A list of control arguments.
#' @export
control_greedy_cluster <- function(
        neighbours = 2L,
        neighbours_type = "regular",
        neighbours_connected = TRUE,
        rng_start = FALSE,
        greedy = 3L,
        greedy_end = FALSE,
        competitive = TRUE,
        competitive_release = 5,
        max_iteration = c(100, 2),
        learning_range = c(1.0, 0.1, 3),
        seed = sample(1e6, 1),
        tolerance = (.Machine$double.eps)^(1/4),
        include_data = TRUE,
        trace = 0L
) {
    ##
    if (!is.infinite(neighbours)) {
        if (!is.numeric(neighbours)) {
            stop("'neighbours' has to be an integer.")
        }
        else {
            neighbours <- as.integer(neighbours)
        }
    }

    if (!(is.character(neighbours_type) & (neighbours_type %in% c("regular", "grid", "hexgrid")))) {
        stop("'neighbours_type' has to be string taken the one of the following values: 'regular', 'grid', or 'hexgrid'.")
    }

    ##
    if (!is.logical(neighbours_connected)) {
        stop("'neighbours_connected' has to be logical.")
    }

    ##
    if (!is.logical(rng_start)) {
        stop("'rng_start' has to be logical.")
    }

    ##
    if (is.null(greedy)) {
        greedy <- 3
    }

    if (!is.numeric(greedy)) {
        stop("'greedy' has to be numeric.")
    }

    if (greedy < 0) {
        stop("'greedy' has to be >= 0.")
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
        competitive_release <- greedy
    }

    if (!is.numeric(competitive_release)) {
        stop("'competitive_release' has to be numeric.")
    }

    ##
    if (!is.numeric(tolerance)) {
        stop("'tolerance' has to be numeric.")
    }

    ##
    if (is.null(include_data)) {
        include_data <- FALSE
    }

    if (!is.logical(include_data)) {
        stop("'include_data' has to be logical.")
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
            neighbours = neighbours,
            neighbours_type = neighbours_type,
            rng_start = rng_start,
            greedy = greedy,
            greedy_end = greedy_end,
            competitive = competitive,
            competitive_release = competitive_release,
            max_iteration = max_iteration,
            learning_range = learning_range,
            tolerance = tolerance,
            seed = seed,
            include_data = include_data,
            trace = trace
        )
    )
}

##
#' @title Greedy K-means Clustering
#'
#' @description Regularised K-means clustering, allowing for greedy re-allocation of data-points when updating clusters, by exploiting the pre-defined neighbourhood structure.
#'
#' @param x Numeric matrix: A matrix with rows and columns corresponding to observations and features, respectively. NB: if \code{x} is not a matrix, the function will try to cast it as a matrix.
#' @param k Integer, integer vector, or adjacency list: The number of clusters, the grid layout, or the adjacency matrix of the clusters, respectively.
#' @param control List: A list of control arguments, for more details see \link{control_greedy_cluster}.
#'
#' @details
#' If the argument \code{k} is given as an adjacency list, then the elements will be taken as the number of clusters and the argument \code{neighbours}, set in the control-object, will be ignored.
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

    ##
    #
    if (!(is.numeric(k) || is.list(k))) {
        stop("'k' should be either an integer or an adjacency list")
    }
    else if (is.numeric(k)) {
        k <- as.integer(k)
    }

    #
    if (!is.integer(k)) {
        #
        if (!is_adjacency_list(k)) {
            stop("'k' should be either an adjacency list.")
        }

        #
        if (control[["neighbours_connected"]]) {
            if (!is_connected_matrix(neighbours)) {
                stop("The adjacency graph should be connected; if not set 'neighbours_connected == FALSE' in control list.")
            }
        }

        #
        neighbours <- lapply(seq_len(length(k)), \(x) which(k[x,] > 0) - 1)
        k <- as.integer(length(k))
    }
    else {
        neighbours <- create_adjacency_list(k, control[["neighbours"]], control[["neighbours_type"]])

        if (length(k) > 1) {
            k <- prod(k)
        }
    }

    ##
    #
    res <- greedy_cluster_cpp(
        x = x,
        k = k,
        neighbourhood = neighbours,
        rng_start = control[["rng_start"]],
        greedy = control[["greedy"]],
        greedy_end = control[["greedy_end"]],
        competitive = control[["competitive"]],
        competitive_release = control[["competitive_release"]],
        lrange = control[["learning_range"]],
        tol = control[["tolerance"]],
        maxiter = control[["max_iteration"]],
        seed = control[["seed"]],
        trace = control[["trace"]]
    )

    # Change from C++ to R indices.
    res[["neighbourhood"]] <- lapply(neighbours, \(x) x + 1)
    res[["previous"]] <- res[["previous"]] + 1

    #
    if (control$include_data) {
        res[["data"]] <- x
    }

    ##
    #
    class(res) <- "cluster"

    #
    return(res)
}


#' @title Cluster predictions
#'
#' @description The allocations predicted by the \link{cluster} object.
#'
#' @param object An object of class \link{cluster}.
#' @param ... Additional arguments.
#'
#' @details A
#'
#' @return A matrix of allocated clusters.
#'
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
            reasign <- FALSE
        }
        else {
            reasign <- TRUE
        }
    }
    else {
        reasign <- dots[["reasign"]]
    }

    if (!is.logical(reasign)) {
        reasign <- as.logical(reasign)
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

