#include <RcppEigen.h>
#include <vector>
#include <cmath>

// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
Rcpp::List conjugate_gradient_cpp(
    const Eigen::MappedSparseMatrix<double>& A,
    const Eigen::Map<Eigen::VectorXd>& b,
    double tol = 1e-6,
    int max_iter = 1000
) {

  int n = b.size();

  Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd r = b;
  Eigen::VectorXd p = r;

  double rs_old = r.squaredNorm();

  std::vector<double> res_log;
  res_log.push_back(std::sqrt(rs_old));

  int k;

  for (k = 0; k < max_iter; k++) {

    if (std::sqrt(rs_old) <= tol) {
      break;
    }

    Eigen::VectorXd Ap = A * p;

    double alpha = rs_old / p.dot(Ap);

    x += alpha * p;
    r -= alpha * Ap;

    double rs_new = r.squaredNorm();
    res_log.push_back(std::sqrt(rs_new));

    double beta = rs_new / rs_old;
    p = r + beta * p;

    rs_old = rs_new;

  }

  return Rcpp::List::create(
    Rcpp::Named("x") = x,
    Rcpp::Named("iterations") = k,
    Rcpp::Named("residual_history") = res_log
  );

}
