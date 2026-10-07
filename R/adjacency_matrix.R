##
#
is_adjacency_list <- function(k, tolerance = 1e-8) {
    ##
    #
    k_clps <- do.call("c", k)
    if (all(abs(k_clps - round(k_clps)) > tolerance)) {
        return(FALSE)
    }

    ##
    #
    if ((min(k) < 1) & (max(k) > length(k))) {
        return(FALSE)
    }

    ##
    #
    return (TRUE)
}

#
create_adjacency_list <- function(k, neighbours, type) {
    if (type == "regular") {
        adj_list <- create_adjacency_list_regular(k, neighbours)
    }
    else if (type == "grid") {
        adj_list <- GKM:::create_adjacency_list_grid(k)
    }
    else if (type == "hexgrid") {
        stop("Currenly the only \"type\"'s implemented are \"regular\" and \"grid\".")
    }
    else {
        stop("Currenly the only \"type\"'s implemented are \"regular\" and \"grid\".")
    }

    return(adj_list)
}

##
#
is_connected_matrix <- function(neighbours, tolerance = 1e-8) {
    ##
    #
    neighbours <- neighbourhood_matrix(neighbours)

    #
    if (all(neighbours == t(neighbours))) {
        d <- diag(apply(neighbours, 1, sum))
        e <- eigen(d - neighbours, only.values = TRUE)[["values"]]

        m <- nrow(neighbours)
        connected <- e[m - 1] > tolerance
    }
    else {
        connected <- is_connected_matrix_directed(adj_matrix)
    }

    #
    return(connected)
}

#
#' @title Create neighbourhood matrix
#'
#' @description Create a neighbourhood matrix from an adjacency list.
#'
#' @param adj_list A list containing the neighbours for each cluster.
#'
#' @return A neighbourhood matrix.
#' @export
neighbourhood_matrix <- function(adj_list) {
    ##
    # Change from R to C++ indices
    adj_list <- adj_list |> lapply(\(x) x - 1)

    ##
    #
    k <- length(adj_list)
    adj_matrix <- neighbourhood_reconstruction(adj_list, k)

    ##
    #
    return(adj_matrix)
}
