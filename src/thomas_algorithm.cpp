#include <RcppEigen.h>
#include <vector>

// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
Eigen::VectorXd thomas_algorithm_cpp(
    const Eigen::VectorXd& a,
    const Eigen::VectorXd& d,
    const Eigen::VectorXd& c,
    const Eigen::VectorXd& b
) {
  int n = b.size();
  Eigen::VectorXd x(n);

  std::vector<double> c_prime(n, 0.0);
  std::vector<double> d_prime(n, 0.0);

  c_prime[0] = c(0) / d(0);
  d_prime[0] = b(0) / d(0);

  for (int i = 1; i < n; i++) {
    double ai = a(i - 1);
    double denom = d(i) - ai * c_prime[i - 1];

    if (i < n - 1) {
      c_prime[i] = c(i) / denom;
    }
    d_prime[i] = (b(i) - ai * d_prime[i - 1]) / denom;
  }

  x(n - 1) = d_prime[n - 1];

  for (int i = n - 2; i >= 0; i--) {
    x(i) = d_prime[i] - c_prime[i] * x(i + 1);
  }

  return x;
}
