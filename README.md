# GKM: Greedy K-means Clustering
## Overview
**GKM** is an R package that implements **greedy re-assignment for K-means clustering**, a two-stage greedy K-means algorithm that leverages Self-Organizing Maps (SOM) to reduce the number of point-to-cluster-centre comparisons. The approach combines batch-SOM with a greedy assignment heuristic, significantly reducing the number of comparisons needed compared to standard K-means.

## Installation
### From GitHub
You can install the development version of **GKM** directly from GitHub using the `remotes` package:

```r
# Install remotes if not already installed
if (!requireNamespace("remotes", quietly = TRUE)) {
    install.packages("remotes")
}

# Install GKM from GitHub
remotes::install_github("svilsen/GKM")
```

## Documentation
For detailed documentation, check out the package vignettes and help files after installation:

```r
# View help for the main function
?GKM
?greedy_cluster
```

## Usage
### Basic Example
Here’s a simple example of how to use the greedy K-means clustering:

```r
# Load the GKM package
library(GKM)

# Generate sample data
x <- iris[, 3:4] |> as.matrix()
cl <- iris[, 5]
k <- length(unigue(cl))

# Run greedy K-means clustering
grd_clst <- greedy_cluster(
    x, 
    k = 3, 
    control = list(
        neighbours = 1L, 
        rng_start = FALSE, 
        greedy = 2L, 
        seed = 123456
    )
)

# Run standard K-means clustering
km_clst <- greedy_cluster(
    x, 
    k = 3, 
    control = list(
        rng_start = FALSE, 
        greedy = 1L, 
        competitive = FALSE,
        seed = 123456
    )
)
```

<img width="2398" height="859" alt="g1" src="https://github.com/user-attachments/assets/bb170cf0-bfa5-4f9c-ba8f-c9cb62cc8455" />

## Contributing
Contributions are welcome! Please feel free to submit issues or pull requests on the [GitHub repository](https://github.com/svilsen/GKM).

## License
This project is licensed under the **MIT License**, see the [LICENSE](https://opensource.org/licenses/MIT) file for details.

## Contact
For questions or feedback, please contact: **Søren B. Vilsen**, [svilsen@mp.aau.dk](mailto:svilsen@mp.aau.dk)

## Acknowledgments
Built with [Rcpp](https://www.rcpp.org/) and [RcppArmadillo](https://github.com/RcppCore/RcppArmadillo).
